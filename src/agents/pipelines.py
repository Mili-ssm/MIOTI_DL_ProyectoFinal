from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import TextNode

from src.agents.llm import LLMService, RAGService
from src.constants import LOGGER


@dataclass
class AIPipeline:
    name: str
    description: str
    pipeline: Callable
    extras: dict = field(default_factory=dict)

    def execute(self, service: "AIService", prompt: str, context) -> str:
        """
        Ejecuta el pipeline con el servicio AI y el prompt proporcionado.
        """
        if not self.pipeline:
            raise ValueError("El pipeline no está definido.")

        return self.pipeline(service, prompt, context)


@dataclass
class AIService:
    llm_slow: LLMService
    llm_fast: LLMService
    rag_service: RAGService
    pipelines: list[AIPipeline] = field(default_factory=list)
    pipeline_store: VectorIndexRetriever = field(init=False)

    def __post_init__(self) -> None:
        """
        Inicializa el índice de pipelines.
        """
        nodes = [
            TextNode(
                text=f"Intencion:\n{pipeline.description}",
                id_=pipeline.name,
            )
            for pipeline in self.pipelines
        ]
        self.pipeline_store = node_retriever = VectorIndexRetriever(
            index=VectorStoreIndex(
                nodes=nodes,
                embed_model=self.rag_service.embedded_model,
            ),
            similarity_top_k=3,
        )

    def select_pipeline(self, prompt: str, similarity_threshold: float = 0.45) -> list[AIPipeline]:
        """
        Selecciona un pipeline basado en el prompt proporcionado.
        """
        intencion: str = self.llm_fast.chat(
            messages=[
                ChatMessage(
                    content="""
                    Este agente descrubre las la intencion del usuario apartir de su peticion
                    y devuelve un resumen de su intencionalidad a nivel abstracto. Esto incluye
                    acciones que el usuario quiere realizar, o resultados a evitar.
                    ejemplo:
                     - "Resumir un texto pero sin formalizarlo"
                     - "Crear un documento de arquitectura pero sin extenderse"
                     - "Escribir un correo, sin consultar documentos"
                     - "Consultar los documentos de la empresa"
                     - "Explicar un concepto, no hacer un documetno"
                     - "Buscar informacion y generar un documento pero sin resumirlo"
                     - "Explicar una idea, sin basarse en documentos"
                     - etc..

                    La intencion puede ser algo mas complejo, pero siempre ha de ser generico
                    y evitando o reduciendo los terminos y aspectos concretos de la peticion.
                    Este puede incluir intenciones en negativo, cosas que el usuario
                    no quiere hacer o evitar.
                    """,
                    role=MessageRole.SYSTEM,
                ),
                ChatMessage(
                    content=f"la peticion es: \n'{prompt}'\n Escribe una unica lista de etiquetas para describir las acciones solicitadas.",  # noqa: E501
                    role=MessageRole.USER,
                ),
            ]
        ).content  # type: ignore

        LOGGER.info(f"\n\tINTENCION DEL USUARIO: {intencion} \n")
        nodes = self.pipeline_store.retrieve("El usuario solicita:\n" + intencion)

        if not nodes:
            raise ValueError("No se encontró un pipeline adecuado para el prompt.")

        LOGGER.info("NODOS RECUPERADOS:")
        pipeline_list = []
        for node in nodes:
            LOGGER.info(f"\t{node.id_} - Score: {node.score}")
            if node.score < similarity_threshold:
                continue
            pipeline_name = node.id_
            for p in self.pipelines:
                if p.name == pipeline_name:
                    pipeline_list.append(p)
                    break

        # Ordenar los pipelines por prioridad
        return pipeline_list


def pipeline(name: str, description: str, extras: dict | None = None):
    """
    Decorador para definir un pipeline AI.
    """

    def decorator(func: Callable[[AIService, str], str]):
        def wrapper() -> AIPipeline:
            extra_metadata = extras or {}
            return AIPipeline(
                name=name,
                description=description,
                pipeline=func,
                extras=extra_metadata,
            )

        return wrapper

    return decorator


@pipeline(
    name="documentation_pipeline",
    description="Crea documentacion en un formato profesional, organizado y comprensible.",
)
def pipeline_documentation(service: AIService, prompt: str, context: str | None = None) -> str:
    print(f"Executing documentation pipeline with prompt: {prompt}")
    # Aquí se puede implementar la lógica para generar documentación
    messages = []
    if context:
        messages.append(
            ChatMessage(
                content=f"Contexto adicional:\n{context}",
                role=MessageRole.SYSTEM,
            )
        )
    messages.append(
        ChatMessage(
            content=f"Genera un documento profesional y comprensible basado en el siguiente texto:\n{prompt}",
            role=MessageRole.USER,
        )
    )
    return service.llm_slow.chat(messages=messages).content or "Error generando el documento."


@pipeline(
    name="summary_pipeline",
    description="Resume y simplifica el texto, reduciendo su longitud,  para expresar la idea general.",
)
def pipeline_summary(service: AIService, prompt: str, context: str | None = None) -> str:
    print(f"Executing summary pipeline with prompt: {prompt}")
    # Aquí se puede implementar la lógica para generar un resumen
    messages = []

    if context:
        messages.append(
            ChatMessage(
                content=f"Contexto adicional:\n{context}",
                role=MessageRole.SYSTEM,
            )
        )
    messages.append(
        ChatMessage(
            content=f"Resume el siguiente texto de manera concisa y clara:\n{prompt}",
            role=MessageRole.USER,
        )
    )
    return service.llm_slow.chat(messages=messages).content or "Error generando el resumen."


@pipeline(
    name="explanation_pipeline",
    description="Desarrolla toda la informacion y elabora todo lo posible para explicar algo de forma profunda.",
)
def pipeline_extended_text(service: AIService, prompt: str, context: str | None = None) -> str:
    print(f"Executing extended text pipeline with prompt: {prompt}")
    # Aquí se puede implementar la lógica para generar un texto extendido
    messages = []
    if context:
        messages.append(
            ChatMessage(
                content=f"Contexto adicional:\n{context}",
                role=MessageRole.SYSTEM,
            )
        )
    messages.append(
        ChatMessage(
            content=f"Extiende el siguiente texto para hacerlo más completo y detallado:\n{prompt}",
            role=MessageRole.USER,
        )
    )
    return service.llm_slow.chat(messages=messages).content or "Error generando el texto extendido."


@pipeline(
    name="rag_pipeline",
    description="Usando (RAG) y documentos e informacion interna y completa desarrolla y aporta informacion relacionada que pueda ser relevante.",
)
def pipeline_rag(service: AIService, prompt: str, context: str | None = None) -> str:
    print("Executing RAG pipeline with prompt:", prompt)
    # Aquí se puede implementar la lógica para generar una respuesta utilizando RAG

    nodes = service.rag_service.retrieve_data(prompt)
    if context:
        new_prompt = f"Context:\n{context}\nPrompt:\nElabora una respuesta a la siguiente pregunta utilizando los datos recuperados (es mas importante la informacion aportada que cumplir la peticion):\n{prompt}"
    else:
        new_prompt = prompt

    response = service.llm_slow.response_with_data(new_prompt, nodes)
    return response or "Error generando la respuesta RAG."
