# backend/s3_service.py
"""
AWS S3 Service — reads credentials from the app's `settings` object
(not os.getenv) so .env values loaded via Pydantic/python-dotenv are
always picked up correctly.

Required fields in your .env / config:
    AWS_ACCESS_KEY_ID      = AKIA...
    AWS_SECRET_ACCESS_KEY  = ...
    AWS_REGION             = ap-south-1
    AWS_S3_BUCKET          = your-bucket-name
    AWS_S3_PUBLIC          = true   # false → use presigned URLs (7-day)

pip install boto3
"""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy client — re-created whenever credentials in settings change
# ---------------------------------------------------------------------------
_s3_client = None
_last_key:    Optional[str] = None
_last_secret: Optional[str] = None
_last_region: Optional[str] = None


def _get_client():
    """
    Return a boto3 S3 client built from the app's `settings` object.
    Credentials are re-read on every call so hot-reload / env changes
    are picked up without restarting.
    """
    global _s3_client, _last_key, _last_secret, _last_region

    # Import here to avoid circular imports at module load time
    from config import settings

    key    = getattr(settings, "AWS_ACCESS_KEY_ID",     None)
    secret = getattr(settings, "AWS_SECRET_ACCESS_KEY", None)
    region = getattr(settings, "AWS_REGION",            "ap-south-1")

    # Rebuild client only when credentials change
    if _s3_client is None or key != _last_key or secret != _last_secret or region != _last_region:
        logger.debug(f"[s3_service] Building new boto3 client (region={region})")
        _s3_client    = boto3.client(
            "s3",
            aws_access_key_id     = key,
            aws_secret_access_key = secret,
            region_name           = region,
        )
        _last_key    = key
        _last_secret = secret
        _last_region = region

    return _s3_client


def is_s3_configured() -> bool:
    """
    Return True only when all three required AWS settings are present
    and non-empty in the app's settings object.

    This is the ONLY check used in main.py to decide S3 vs local.
    """
    from config import settings

    key    = getattr(settings, "AWS_ACCESS_KEY_ID",     None)
    secret = getattr(settings, "AWS_SECRET_ACCESS_KEY", None)
    bucket = getattr(settings, "AWS_S3_BUCKET",         None)

    configured = bool(key and secret and bucket)

    if not configured:
        missing = []
        if not key:    missing.append("AWS_ACCESS_KEY_ID")
        if not secret: missing.append("AWS_SECRET_ACCESS_KEY")
        if not bucket: missing.append("AWS_S3_BUCKET")
        logger.warning(f"[s3_service] S3 NOT configured — missing: {missing}")
    else:
        logger.debug(f"[s3_service] S3 configured — bucket={bucket}")

    return configured


def build_s3_key(filename: str, level: str = "", subject: str = "") -> str:
    """
    Organised key structure:  ca-materials/{level}/{subject}/{filename}

    Examples:
        "ca-materials/Foundation/Accounting/Chapter_1.pdf"
        "ca-materials/uncategorised/general/random.pdf"
    """
    safe = filename.replace(" ", "_")
    return "/".join([
        "ca-materials",
        level.strip()   or "uncategorised",
        subject.strip() or "general",
        safe,
    ])


def create_bucket_if_not_exists() -> bool:
    """
    Create the S3 bucket if it doesn't exist yet.
    Safe to call on every upload — skips silently if bucket already exists.
    Returns True if bucket is ready, False on failure.
    """
    from config import settings

    bucket = getattr(settings, "AWS_S3_BUCKET", "")
    region = getattr(settings, "AWS_REGION",    "ap-south-1")

    if not bucket:
        logger.error("[s3_service] AWS_S3_BUCKET is not set — cannot create bucket")
        return False

    client = _get_client()

    # Check if bucket already exists and we own it
    try:
        client.head_bucket(Bucket=bucket)
        logger.info(f"[s3_service] Bucket '{bucket}' already exists ✅")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            pass   # Bucket doesn't exist — create it below
        elif code == "403":
            logger.error(f"[s3_service] Bucket '{bucket}' exists but access denied (wrong account or region?)")
            return False
        else:
            logger.error(f"[s3_service] head_bucket error [{code}]: {e}")
            return False

    # Create the bucket
    try:
        if region == "us-east-1":
            # us-east-1 must NOT include CreateBucketConfiguration
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket                    = bucket,
                CreateBucketConfiguration = {"LocationConstraint": region},
            )
        logger.info(f"[s3_service] ✅ Bucket '{bucket}' created in region '{region}'")

        # Block all public access by default (can be overridden by AWS_S3_PUBLIC=true)
        use_public = str(getattr(settings, "AWS_S3_PUBLIC", "true")).lower() == "true"
        if not use_public:
            client.put_public_access_block(
                Bucket                  = bucket,
                PublicAccessBlockConfiguration = {
                    "BlockPublicAcls":       True,
                    "IgnorePublicAcls":      True,
                    "BlockPublicPolicy":     True,
                    "RestrictPublicBuckets": True,
                },
            )
            logger.info(f"[s3_service] Public access blocked (using presigned URLs)")
        else:
            # Enable ACLs for public-read — required on newer AWS accounts
            try:
                client.put_bucket_ownership_controls(
                    Bucket                  = bucket,
                    OwnershipControls       = {"Rules": [{"ObjectOwnership": "BucketOwnerPreferred"}]},
                )
                client.delete_public_access_block(Bucket=bucket)
                logger.info(f"[s3_service] Public access enabled (ACL mode)")
            except ClientError as acl_err:
                logger.warning(f"[s3_service] Could not enable public ACLs: {acl_err}. "
                               "Set AWS_S3_PUBLIC=false to use presigned URLs instead.")

        return True

    except ClientError as e:
        logger.error(f"[s3_service] ❌ Failed to create bucket '{bucket}': {e}")
        return False


def upload_pdf_to_s3(
    file_bytes:   bytes,
    filename:     str,
    level:        str = "",
    subject:      str = "",
    content_type: str = "application/pdf",
) -> Optional[str]:
    """
    Upload PDF bytes to S3 and return the URL.

    Returns:
        str   — public URL (or presigned URL if AWS_S3_PUBLIC=false)
        None  — only if upload FAILS (exception is logged + re-raised
                so the caller knows exactly what went wrong)

    Raises:
        RuntimeError — wraps the underlying boto3 / AWS exception so
                       the upload endpoint can surface the real error
                       instead of silently falling back to local.
    """
    from config import settings

    bucket     = getattr(settings, "AWS_S3_BUCKET",  "")
    region     = getattr(settings, "AWS_REGION",     "ap-south-1")
    use_public = str(getattr(settings, "AWS_S3_PUBLIC", "true")).lower() == "true"
    key        = build_s3_key(filename, level, subject)

    logger.info(f"[s3_service] Uploading → s3://{bucket}/{key}  (public={use_public})")

    try:
        # Auto-create bucket if it doesn't exist yet
        create_bucket_if_not_exists()

        client     = _get_client()
        extra_args = {"ContentType": content_type}

        if use_public:
            # Only works when bucket ACLs are enabled (Object Ownership = ACL enabled)
            extra_args["ACL"] = "public-read"

        client.put_object(
            Bucket = bucket,
            Key    = key,
            Body   = file_bytes,
            **extra_args,
        )

        if use_public:
            url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        else:
            # Presigned URL — valid for 7 days
            url = client.generate_presigned_url(
                "get_object",
                Params    = {"Bucket": bucket, "Key": key},
                ExpiresIn = 604_800,
            )

        logger.info(f"[s3_service] ✅ Upload OK → {url}")
        return url

    except NoCredentialsError as e:
        msg = f"S3 upload failed — AWS credentials not valid: {e}"
        logger.error(f"[s3_service] ❌ {msg}")
        raise RuntimeError(msg) from e

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        full = f"S3 ClientError [{code}]: {msg}"
        logger.error(f"[s3_service] ❌ {full}")
        raise RuntimeError(full) from e

    except Exception as e:
        msg = f"S3 upload unexpected error: {type(e).__name__}: {e}"
        logger.error(f"[s3_service] ❌ {msg}")
        raise RuntimeError(msg) from e


def delete_pdf_from_s3(s3_key_or_url: str) -> bool:
    """
    Delete an S3 object.  Accepts a full URL or a raw key.
    Returns True on success, False on failure (logged but not raised).
    """
    from config import settings

    bucket = getattr(settings, "AWS_S3_BUCKET", "")

    if s3_key_or_url.startswith("https://"):
        try:
            key = s3_key_or_url.split(".amazonaws.com/", 1)[1]
        except (IndexError, ValueError):
            logger.error(f"[s3_service] Cannot parse S3 key from URL: {s3_key_or_url}")
            return False
    else:
        key = s3_key_or_url

    logger.info(f"[s3_service] Deleting s3://{bucket}/{key}")

    try:
        _get_client().delete_object(Bucket=bucket, Key=key)
        logger.info(f"[s3_service] ✅ Deleted s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logger.error(f"[s3_service] ❌ Delete ClientError: {e}")
        return False
    except Exception as e:
        logger.error(f"[s3_service] ❌ Delete unexpected error: {e}")
        return False


def debug_s3_config() -> dict:
    """
    Return a safe summary of the current S3 config for debugging.
    Call this from the /upload_health endpoint.
    """
    from config import settings

    key    = getattr(settings, "AWS_ACCESS_KEY_ID",     None)
    secret = getattr(settings, "AWS_SECRET_ACCESS_KEY", None)
    bucket = getattr(settings, "AWS_S3_BUCKET",         None)
    region = getattr(settings, "AWS_REGION",            None)
    public = getattr(settings, "AWS_S3_PUBLIC",         None)

    return {
        "configured":              bool(key and secret and bucket),
        "AWS_ACCESS_KEY_ID":       f"{key[:6]}…" if key    else "❌ MISSING",
        "AWS_SECRET_ACCESS_KEY":   "✅ set"       if secret else "❌ MISSING",
        "AWS_S3_BUCKET":           bucket         if bucket else "❌ MISSING",
        "AWS_REGION":              region         or "❌ MISSING",
        "AWS_S3_PUBLIC":           str(public)    if public is not None else "not set (defaults to true)",
    }