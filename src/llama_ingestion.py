from dataclasses import dataclass

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from src.constants import (
    DB_NAME,
    DOCUMENTS_PATH,
    INDEX_NAME,
    MONGO_COLLECTION,
    MONGO_HOST,
    OPENAI_KEY,
)


@dataclass
class IngestionPipelineConfig:
    documents_path: str = DOCUMENTS_PATH
    mongo_uri: str = MONGO_HOST
    db_name: str = DB_NAME
    collection: str = MONGO_COLLECTION
    index_name: str = INDEX_NAME

    def run_pipeline(self) -> VectorStoreIndex:

        documents = SimpleDirectoryReader(self.documents_path, recursive=True).load_data()

        client = MongoClient(self.mongo_uri)
        vector_store = MongoDBAtlasVectorSearch(
            client=client,
            db_name=self.db_name,
            collection_name=self.collection,
            index_name=self.index_name,
        )

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=25, chunk_overlap=0),
                TitleExtractor(),
                OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_KEY),
            ],
            vector_store=vector_store,
        )

        pipeline.run(documents=documents)

        return VectorStoreIndex.from_vector_store(vector_store)
