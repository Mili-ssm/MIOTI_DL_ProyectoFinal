import logging
import os

from dotenv import load_dotenv

load_dotenv()

MONGO_HOST: str = os.getenv("MONGO_HOST", "mongodb+srv://localhost:27017")

DB_NAME: str = os.getenv("DB_NAME", "rag_db")

MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "rag_collection")

INDEX_NAME: str = os.getenv("INDEX_NAME", "vector_index")

EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))
"""Dimensión de embedding para OpenAI y LlamaIndex"""

DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "../Document")

HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")

OPENAI_KEY: str = os.getenv("OPENAI_KEY", "")


LOGGER = logging.getLogger(__name__)
