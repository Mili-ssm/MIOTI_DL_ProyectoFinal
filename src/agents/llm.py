from dataclasses import dataclass

import torch
from llama_index.core import get_response_synthesizer
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from zmq import Enum

from src.constants import HUGGINGFACE_TOKEN, OPENAI_KEY


class Provider(Enum):
    OPENAI = "openai"
    LLAMA = "llama"


@dataclass
class LLMConfig:
    provider: Provider = Provider.LLAMA
    llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"

    def get_agent(self) -> LLM:
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
class Agent:
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
    def from_config(cls, config: LLMConfig) -> "Agent":
        """
        Crea una instancia de LLMS a partir de la configuración proporcionada.
        """
        llm = config.get_agent()
        return cls(llm=llm)
