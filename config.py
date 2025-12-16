from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB: str = "ca_chatbot"

    JWT_SECRET: str
    JWT_ALGO: str = "HS256"

    PINECONE_API_KEY: str
    PINECONE_INDEX: str  # must already exist in Pinecone dashboard

    OPENROUTER_API_KEY: str
    LLM_MODEL: str = "openai/gpt-4.1-mini"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

    FRONTEND_ORIGIN: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()
