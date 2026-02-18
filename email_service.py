import smtplib
from email.message import EmailMessage
from config import settings


def send_admin_signup_notification(user_data: dict):
    msg = EmailMessage()
    msg["Subject"] = "New Student Signup"
    msg["From"] = settings.EMAIL_FROM
    msg["To"] = settings.ADMIN_EMAIL

    msg.set_content(f"""
New student signup:

Name: {user_data['name']}
Email: {user_data['email']}
Phone: {user_data['phone']}
CA Level: {user_data['ca_level']}
Attempt: {user_data['ca_attempt']}

Please approve from admin dashboard.
""")

    with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
        server.starttls()
        server.login(settings.EMAIL_USERNAME, settings.EMAIL_PASSWORD)
        server.send_message(msg)
