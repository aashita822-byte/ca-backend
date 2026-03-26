from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    MONGO_URI: str
    MONGO_DB: str = "ca_chatbot"

    # Auth
    JWT_SECRET: str
    JWT_ALGO: str = "HS256"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str

    # OpenAI
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4.1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Frontend
    FRONTEND_ORIGIN: str = "*"

    # Email
    EMAIL_HOST: str
    EMAIL_PORT: int = 587
    EMAIL_USERNAME: str
    EMAIL_PASSWORD: str
    EMAIL_FROM: str
    ADMIN_EMAIL: str

    # Razorpay
    RAZORPAY_KEY_ID: str
    RAZORPAY_KEY_SECRET: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
