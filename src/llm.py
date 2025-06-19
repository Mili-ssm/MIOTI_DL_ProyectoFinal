from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import certifi
import torch
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

from constants import (
    DB_NAME,
    HUGGINGFACE_TOKEN,
    INDEX_NAME,
    MONGO_COLLECTION,
    MONGO_HOST,
    OPENAI_KEY,
)

if TYPE_CHECKING:
    from llama_index.core.schema import Document


class Provider(Enum):
    OPENAI = "openai"
    LLAMA = "llama"


@dataclass(init=False)
class RAGService:
    vector_store: MongoDBAtlasVectorSearch
    splitter: SentenceSplitter
    index: VectorStoreIndex

    def __init__(self, spliter_size: int, spliter_overlap: int, llm, embedding) -> None:
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
        )

        self.splitter = SentenceSplitter(chunk_size=spliter_size, chunk_overlap=spliter_overlap)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=embedding,
            show_progress=True,
        )


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
            self.embedding = OpenAIEmbedding(
                model=embed_model_name or "text-embedding-3-small",
                api_key=OPENAI_KEY,
            )
            self.llm = OpenAI(
                model=llm_model_name or "gpt-4o-mini",
                api_key=OPENAI_KEY,
            )
        elif provider == Provider.LLAMA:
            print(embed_model_name)
            print(llm_model_name)
            self.embedding = HuggingFaceEmbedding(
                model_name=embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = HuggingFaceLLM(
                tokenizer_name=llm_model_name or "meta-llama/Llama-3.2-3B-Instruct",
                model_name=llm_model_name or "meta-llama/Llama-3.2-3B-Instruct",
                # context_window=8192,
                max_new_tokens=256,
                model_kwargs={
                    "token": HUGGINGFACE_TOKEN,
                    "torch_dtype": torch.bfloat16,
                },
                tokenizer_kwargs={
                    "token": HUGGINGFACE_TOKEN,
                    "torch_dtype": torch.bfloat16,
                    "use_fast": True,
                },
            )
        self.rag = RAGService(
            spliter_size=512,
            spliter_overlap=64,
            llm=self.llm,
            embedding=self.embedding,
        )

    # .................................................................
    # Consultas
    # .................................................................
    def query(self, question: str, top_k: int = 3) -> str:
        """Realiza consulta RAG simple y devuelve respuesta sintetizada."""
        # Construimos retriever y synthesizer por defecto --------------

        # Instantiate Atlas Vector Search as a retriever
        retriever = self.rag.index.as_retriever(similarity_top_k=top_k, verbose=True)
        retrieved_nodes = retriever.retrieve(question)

        print(f"Nodes recuperados: {len(retrieved_nodes)}")

        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,
        )
        response = response_synthesizer.synthesize(
            query=question,
            nodes=retrieved_nodes,
        )

        return str(response.response)

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
    llm_service = LLMService(provider=Provider.LLAMA)
    questions = [
        "con que sitios colaboramos?",
        "SDP Technology Architecture, whats next?",
        "What links there are between the documents?",
    ]
    for question in questions:
        response = llm_service.query(question)
        print(f"Pregunta: {question}")
        print(f"Respuesta: {response}")
# ---------------------------------------------------------------------------
