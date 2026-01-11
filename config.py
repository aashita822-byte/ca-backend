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
    PINECONE_INDEX: str  # must exist in Pinecone dashboard

    # OpenAI (DIRECT API)
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Frontend
    FRONTEND_ORIGIN: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
