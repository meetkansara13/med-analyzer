from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "AI Medical Document Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HUGGINGFACE_TOKEN: str = ""
    NER_MODEL: str = "d4data/biomedical-ner-all"
    SUMMARIZER_MODEL: str = "Falconsai/medical_summarization"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "medical_docs"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT: str = "medical-analyzer"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
