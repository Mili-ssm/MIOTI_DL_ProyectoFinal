from collections.abc import Sequence
from dataclasses import dataclass
from math import e
from typing import TYPE_CHECKING

import certifi
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
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

if TYPE_CHECKING:
    from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding


@dataclass
class IngestionPipelineConfig:
    llm: LLM
    chunk_size: int = 512
    chunk_overlap: int = 10
    documents_path: str = DOCUMENTS_PATH
    mongo_uri: str = MONGO_HOST
    db_name: str = DB_NAME
    collection: str = MONGO_COLLECTION
    index_name: str = INDEX_NAME

    def run_pipeline(self, embedding: "MultiModalEmbedding") -> Sequence[BaseNode]:
        documents = SimpleDirectoryReader(self.documents_path, recursive=True).load_data()

        if self.mongo_uri:
            client = MongoClient(self.mongo_uri, tlsCAFile=certifi.where())
            vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=client,
                db_name=self.db_name,
                collection_name=self.collection,
                vector_index_name=self.index_name,
            )
        else:
            vector_store = None

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
                TitleExtractor(llm=self.llm),
                embedding,
            ],
            vector_store=vector_store,
        )

        return pipeline.run(documents=documents)
