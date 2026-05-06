import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
        self.GEMINI_EMBEDDING_MODEL  = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001"
        )


settings = Settings()