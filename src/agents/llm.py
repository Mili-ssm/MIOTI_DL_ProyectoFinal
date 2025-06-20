from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Self

import certifi
import torch
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    from llama_index.core.base.response.schema import Response
    from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding


class Provider(Enum):
    OPENAI = "openai"
    LLAMA = "llama"


@dataclass
class LLMConfig:
    provider: Provider = Provider.LLAMA
    llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"

    def get_agent_model(self) -> LLM:
        """
        Devuelve el nombre del modelo LLM según el proveedor.
        """
        match self.provider:
            case Provider.OPENAI:
                return OpenAI(
                    model=self.llm_model_name,
                    api_key=OPENAI_KEY,
                )
            case Provider.LLAMA:
                return HuggingFaceLLM(
                    tokenizer_name=self.llm_model_name,
                    model_name=self.llm_model_name,
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
            case _:
                raise ValueError(f"Proveedor LLM no soportado: {self.provider}")


@dataclass
class RAGConfig:
    """
    Configuración para el agente RAG.
    """

    embedding_model_name: str = "all-MiniLM-L6-v2"

    def get_embedded_model(self) -> "MultiModalEmbedding":
        """
        Devuelve una instancia de RAGService configurada.
        """
        embedded_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
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
        return embedded_model  # noqa: RET504


@dataclass
class RAGService:
    vector_stores: VectorStoreIndex
    node_retriver: VectorIndexRetriever

    def retrieve_data(self, query: str) -> list[NodeWithScore]:
        """
        Recupera nodos relevantes del vector store para la consulta dada.
        """
        nodes = self.node_retriver.retrieve(query)
        return nodes  # noqa: RET504

    @classmethod
    def from_config(cls, config: RAGConfig) -> Self:
        """
        Crea una instancia de RAGService a partir de la configuración proporcionada.
        """
        mongo_vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=MongoClient(MONGO_HOST, tlsCAFile=certifi.where()),
            db_name=DB_NAME,
            collection_name=MONGO_COLLECTION,
            vector_index_name=INDEX_NAME,
        )
        vector_store = VectorStoreIndex(
            nodes=[],
            storage_context=StorageContext.from_defaults(
                vector_store=mongo_vector_store,
            ),
            embed_model=config.get_embedded_model(),
        )
        node_retriever = VectorIndexRetriever(
            index=vector_store,
            similarity_top_k=3,
        )
        return cls(vector_stores=vector_store, node_retriver=node_retriever)


@dataclass
class LLMService:
    llm: LLM

    def response_with_data(self, prompt: str, nodes: list[NodeWithScore]) -> str:
        """
        Genera una respuesta del LLM utilizando el prompt y los nodos proporcionados.
        """
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,
        )
        response: Response = response_synthesizer.synthesize(  # type: ignore
            query=prompt,
            nodes=nodes,
        )
        return str(response.response)

    def complete(self, prompt: str) -> str:
        """
        Completa el prompt utilizando el modelo LLM configurado.
        """
        response = self.llm.complete(prompt=prompt)
        return response.text

    def chat(self, messages: list[ChatMessage]) -> ChatMessage:
        """
        Realiza una conversación con el modelo LLM utilizando los mensajes proporcionados.
        """
        response = self.llm.chat(messages=messages)
        return response.message

    @classmethod
    def from_config(cls, config: LLMConfig) -> Self:
        """
        Crea una instancia de LLMS a partir de la configuración proporcionada.
        """
        llm = config.get_agent_model()
        return cls(llm=llm)
