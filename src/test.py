import os
from collections.abc import Iterable, Sequence
from pathlib import Path
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

if TYPE_CHECKING:
    from llama_index.core.schema import Document


EMBEDDING_DIM = 1536  # Dimensión de embedding para OpenAI y LlamaIndex


# ---------------------------------------------------------------------------
# Clases y funciones comunes
# ---------------------------------------------------------------------------


class RAGService:
    """Servicio RAG parametrizable para provider *llama* u *openai*."""

    def __init__(
        self,
        provider: str,
        mongo_uri: str,
        db_name: str,
        collection: str,
        index_name: str = "vector_index",
        embed_model_name: str | None = None,
        llm_model_name: str | None = None,
        **llm_kwargs,
    ) -> None:
        self.provider = provider.lower()
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection = collection
        self.index_name = index_name
        self.client = MongoClient(mongo_uri)

        # Selección de modelos ------------------------------------------------
        if self.provider == "openai":
            self.embed_model = OpenAIEmbedding(model=embed_model_name or "text-embedding-3-small")
            self.llm = OpenAI(model=llm_model_name or "gpt-4o", **llm_kwargs)
        elif self.provider == "llama":
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = HuggingFaceLLM(
                model_name=llm_model_name or "meta-llama/Meta-Llama-3-8B-Instruct", **llm_kwargs
            )
        else:
            raise ValueError("provider must be 'openai' or 'llama'")

        # Vector store --------------------------------------------------------
        self.vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.client,
            db_name=self.db_name,
            collection_name=self.collection,
            index_name=self.index_name,
            embed_dim=EMBEDDING_DIM,
        )

        # Contexto de almacenamiento para LlamaIndex -------------------------
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Splitter por oraciones para índices pequeños; sustituir en producción
        self.splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    # .................................................................
    # Ingesta de documentos
    # .................................................................
    def ingest_paths(self, paths: Sequence[str | Path]) -> None:
        """Lee rutas (patrones glob) y sube documentos al vector store."""
        docs: list[Document] = []
        for pattern in paths:
            for p in Path().glob(str(pattern)):
                reader = SimpleDirectoryReader(input_files=[str(p)])
                docs.extend(reader.load_data())
        # Fragmentamos para granularidad fina --------------------------
        nodes = self.splitter.get_nodes_from_documents(docs)
        self.storage_context.docstore.add_documents(nodes)  # docstore opc.
        self.vector_store.add(nodes)  # vector index

    # .................................................................
    # Consultas
    # .................................................................
    def query(self, question: str, top_k: int = 4) -> str:
        """Realiza consulta RAG simple y devuelve respuesta sintetizada."""
        index = VectorStoreIndex(
            [],
            storage_context=self.storage_context,
            embed_model=self.embed_model,
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

    # -----------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------
    def clear_index(self) -> None:
        """Elimina todos los vectores del índice."""
        self.client[self.db_name][self.collection].delete_many({})


# ---------------------------------------------------------------------------
# Flujos *Proof-of-Concept* (nivel alto) ------------------------------------
# ---------------------------------------------------------------------------


def build_and_query_poc(
    provider: str,
    mongo_uri: str,
    db_name: str,
    collection: str,
    paths: Sequence[str | Path],
    question: str,
    **service_kwargs,
) -> str:
    """Función *todo-en-uno* para pruebas rápidas de RAG (POC).

    1. **Crea** un ``RAGService``.
    2. **Ingresa** los documentos indicados.
    3. **Devuelve** la respuesta a la pregunta.
    """
    svc = RAGService(
        provider=provider,
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection=collection,
        **service_kwargs,
    )
    svc.ingest_paths(paths)
    return svc.query(question)


# ---------------------------------------------------------------------------
# Módulo de test rápido -----------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(description="Quick RAG test script")
    parser.add_argument("question", help="Pregunta a realizar")
    parser.add_argument("--provider", default="openai", choices=["openai", "llama"])
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGODB_URI", ""))
    parser.add_argument("--db", default="ragdb")
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--paths", nargs="+", default=["./data/*.md"])

    args = parser.parse_args()

    answer = build_and_query_poc(
        provider=args.provider,
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        collection=args.collection,
        paths=args.paths,
        question=args.question,
    )
    print(textwrap.fill(answer, width=88))
