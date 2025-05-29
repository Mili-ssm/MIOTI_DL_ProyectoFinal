from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from regex import R

if TYPE_CHECKING:
    from llama_index.core.schema import Document


MONGO_HOST: str = "mongodb://localhost:27017"
DB_NAME: str = "rag_db"
MONGO_COLLECTION: str = "rag_collection"
INDEX_NAME: str = "vector_index"
EMBEDDING_DIM: int = 1536  # Dimensión de embedding para OpenAI y LlamaIndex


class Provider(Enum):
    OPENAI = "openai"
    LLAMA = "llama"


@dataclass(init=False)
class RAGService:
    vector_store: MongoDBAtlasVectorSearch
    context_store: StorageContext
    splitter: SentenceSplitter

    def __init__(self, spliter_size: int, spliter_overlap: int) -> None:
        self.vector_store = MongoDBAtlasVectorSearch(
            mongo_client=MongoClient(MONGO_HOST),
            db_name=DB_NAME,
            collection_name=MONGO_COLLECTION,
            index_name=INDEX_NAME,
            embedding_dim=EMBEDDING_DIM,
        )
        self.context_store = StorageContext.from_defaults(vector_store=self.vector_store)
        self.splitter = SentenceSplitter(chunk_size=spliter_size, chunk_overlap=spliter_overlap)


@dataclass(init=False)
class LLMService:
    llm: HuggingFaceLLM | OpenAI
    rag: RAGService
    embedding: HuggingFaceEmbedding | OpenAIEmbedding

    def __init__(
        self,
        provider: Provider,
        llm_model_name: str,
        embed_model_name: str,
    ) -> None:
        if provider == Provider.OPENAI:
            self.embedding = OpenAIEmbedding(model=embed_model_name or "text-embedding-3-small")
            self.llm = OpenAI(model=llm_model_name or "gpt-4o")
        elif provider == Provider.LLAMA:
            self.embedding = HuggingFaceEmbedding(
                model_name=embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = HuggingFaceLLM(
                model_name=llm_model_name or "meta-llama/Meta-Llama-3-8B-Instruct"
            )
        self.rag_config = RAGService(spliter_size=512, spliter_overlap=64)

    # .................................................................
    # Consultas
    # .................................................................
    def query(self, question: str, top_k: int = 4) -> str:
        """Realiza consulta RAG simple y devuelve respuesta sintetizada."""
        index = VectorStoreIndex(
            [],
            storage_context=self.rag.context_store,
            embed_model=self.embedding,
            llm=self.llm,
        )
        # Construimos retriever y synthesizer por defecto --------------

        response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
        retriever = index.as_retriever(similarity_top_k=top_k)

        retrieved_nodes = retriever.retrieve(question)

        response = response_synthesizer.synthesize(
            query_str=question,
            nodes=retrieved_nodes,
            llm=self.llm,
        )

        return str(response)

    # -----------------------------------------------------------------
    # Sección AGENTIC (stubs para extender)
    # -----------------------------------------------------------------
    def plan_and_answer_agt(self, question: str) -> str:
        """Pipeline agentico: ej. RAG-Fusion, re-ranking, herramientas externas.

        *Stub* — Implementar planificación y pasos adicionales.
        """
        # Ejemplo mínimo: RAG + self-reflection
        initial_answer = self.query(question)
        critique = self.llm.complete(
            f"""Actúa como crítico.
            Pregunta: {question}
            Respuesta inicial: {initial_answer}
            Evalúa exhaustividad y señala mejoras.
            """
        )
        improved_answer = self.llm.complete(
            f"""Mejora la respuesta considerando el siguiente feedback:
            {critique}
            """
        )
        return str(improved_answer)
