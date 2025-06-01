import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import certifi
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface.utils import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()

if TYPE_CHECKING:
    from llama_index.core.schema import Document


MONGO_HOST: str = os.getenv("MONGO_HOST", "mongodb+srv://localhost:27017")
DB_NAME: str = os.getenv("DB_NAME", "rag_db")
MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "rag_collection")
INDEX_NAME: str = os.getenv("INDEX_NAME", "vector_index")
EMBEDDING_DIM: int = int(
    os.getenv("EMBEDDING_DIM", "1536")
)  # Dimensión de embedding para OpenAI y LlamaIndex


class Provider(Enum):
    OPENAI = "openai"
    LLAMA = "llama"


@dataclass(init=False)
class RAGService:
    vector_store: MongoDBAtlasVectorSearch
    context_store: StorageContext
    splitter: SentenceSplitter

    def __init__(self, spliter_size: int, spliter_overlap: int) -> None:
        print("Inicializando servicio RAG...")
        print(f"Conectando a MongoDB en {MONGO_HOST}...")
        print(
            f"Usando base de datos: {DB_NAME}, colección: {MONGO_COLLECTION}, índice: {INDEX_NAME}"
        )
        mongo_client = MongoClient(MONGO_HOST, tlsCAFile=certifi.where())
        self.vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=mongo_client,
            db_name=DB_NAME,
            collection_name=MONGO_COLLECTION,
            vector_index_name=INDEX_NAME,
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
        llm_model_name: str | None = None,
        embed_model_name: str | None = None,
    ) -> None:
        if provider == Provider.OPENAI:
            self.embedding = OpenAIEmbedding(model=embed_model_name or "text-embedding-3-small")
            self.llm = OpenAI(model=llm_model_name or "gpt-4o")
        elif provider == Provider.LLAMA:
            print(embed_model_name)
            print(llm_model_name)
            self.embedding = HuggingFaceEmbedding(
                model_name=embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = HuggingFaceLLM(
                model_name=llm_model_name or "meta-llama/Meta-Llama-3-8B-Instruct"
            )
        self.rag = RAGService(spliter_size=512, spliter_overlap=64)

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

        retriever = index.as_retriever(similarity_top_k=top_k)

        retrieved_nodes = retriever.retrieve(question)

        print(f"Nodes recuperados: {len(retrieved_nodes)}")
        print(f"Primer nodo: {retrieved_nodes}...")  # Muestra los primeros 100 caracteres

        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,
        )
        response = response_synthesizer.synthesize(
            query=question,
            nodes=retrieved_nodes,
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


if __name__ == "__main__":
    # ejemplo con llama
    rag_service = RAGService(spliter_size=512, spliter_overlap=64)
    llm_service = LLMService(
        provider=Provider.LLAMA,
        llm_model_name="AgentPublic/llama3-instruct-8b",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    question = "con que sitios colaboramos?"
    response = llm_service.query(question)
    print(f"Pregunta: {question}")
    print(f"Respuesta: {response}")
# ---------------------------------------------------------------------------
