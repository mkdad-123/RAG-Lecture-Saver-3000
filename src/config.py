# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str
    embedding_model_name: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss.index"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()