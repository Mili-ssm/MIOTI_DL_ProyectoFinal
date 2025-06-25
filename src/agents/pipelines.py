from collections.abc import Callable
from dataclasses import dataclass, field

from llama_cloud import MessageRole, Role
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.schema import TextNode

from src.agents.llm import LLMService, RAGService
from src.constants import LOGGER


@dataclass
class AIPipeline:
    name: str
    description: str
    pipeline: Callable

    def execute(self, service: "AIService", prompt: str) -> str:
        """
        Ejecuta el pipeline con el servicio AI y el prompt proporcionado.
        """
        if not self.pipeline:
            raise ValueError("El pipeline no está definido.")

        return self.pipeline(service, prompt)


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
                text=f"{pipeline.name}: {pipeline.description}",
                id_=pipeline.name,
            )
            for pipeline in self.pipelines
        ]
        self.pipeline_store = node_retriever = VectorIndexRetriever(
            index=VectorStoreIndex(
                nodes=nodes,
                embed_model=self.rag_service.embedded_model,
            ),
            similarity_top_k=1,
        )

    def select_pipeline(self, prompt: str) -> AIPipeline:
        """
        Selecciona un pipeline basado en el prompt proporcionado.
        """
        intencion = self.llm_fast.chat(
            messages=[
                ChatMessage(
                    content="""
                    Este agente descrubre las etiquetas de la peticion del usuario,
                    las etiquetas son una serie de verbos que describen las acciones
                    que el usuario desea realiazar, las cuales se almacenaran en la DB.
                    Ejemplos de posibles resultados:
                        'extender'
                        'resumir'
                        'consultar / resumir'
                        'extender / formalizar'
                        'crear-documento / extender'
                        'consultar / resumir / crear-documento / formalizar'
                        'consultar / crear-documento'
                        'consultar'
                        ...
                    Hay muchas etiquetas y pueden ser combinadas, pero siempre
                    seran una combinacion de verbos (acciones) unicos.
                    """,
                    role=MessageRole.SYSTEM,
                ),
                ChatMessage(
                    content=f"la peticion es: \n'{prompt}'\n Escribe una unica lista de etiquetas para describir las acciones solicitadas.",
                    role=MessageRole.USER,
                ),
            ]
        ).content

        LOGGER.info(f"\n\tINTENCION DEL USUARIO: {intencion} \n")
        node = self.pipeline_store.retrieve(intencion)

        if not node:
            raise ValueError("No se encontró un pipeline adecuado para el prompt.")

        pipeline_name = node[0].node.id_
        for pipeline in self.pipelines:
            if pipeline.name == pipeline_name:
                return pipeline

        raise ValueError(f"Pipeline '{pipeline_name}' no encontrado en la lista de pipelines.")


def pipeline(name: str, description: str):
    """
    Decorador para definir un pipeline AI.
    """

    def decorator(func: Callable[[AIService, str], str]):
        def wrapper() -> AIPipeline:
            return AIPipeline(name=name, description=description, pipeline=func)

        return wrapper

    return decorator


@pipeline(name="test_pipeline", description="A simple test pipeline.")
def pipeline_test(service: AIService, prompt: str) -> str:  # noqa: ARG001
    """Un pipeline de prueba que simplemente devuelve el prompt recibido."""
    print(f"Executing test pipeline with prompt: {prompt}")
    return f"Test pipeline executed with prompt: {prompt}"


@pipeline(
    name="documentation_pipeline",
    description="Genera documentacion basada en el texto para lograr un documento mas profesional y comprensible.",
)
def pipeline_documentation(service: AIService, prompt: str) -> str:  # noqa: ARG001
    """Genera documentación basada en el prompt recibido."""
    print(f"Executing documentation pipeline with prompt: {prompt}")
    # Aquí se puede implementar la lógica para generar documentación
    return "Documentation generated "


@pipeline(
    name="summary_pipeline",
    description="Genera un resumen basado en el texto para reducir su contenido.",
)
def pipeline_summary(service: AIService, prompt: str) -> str:  # noqa: ARG001
    """Genera un resumen basado en el prompt recibido."""
    print(f"Executing summary pipeline with prompt: {prompt}")
    # Aquí se puede implementar la lógica para generar un resumen
    return "Summary generated"


@pipeline(
    name="extended_text_pipeline",
    description="Genera un texto extenso y completo apartir de un concepto simple o reducido.",
)
def pipeline_extended_text(service: AIService, prompt: str) -> str:  # noqa: ARG001
    """Genera un texto extendido basado en el prompt recibido."""
    print(f"Executing extended text pipeline with prompt: {prompt}")
    # Aquí se puede implementar la lógica para generar un texto extendido
    return "Extended text generated"


@pipeline(name="rag_pipeline", description="Generates a response using local Documentation.")
def pipeline_rag(service: AIService, prompt: str) -> str:
    """Genera una respuesta utilizando RAG basada en el prompt recibido."""
    print("Executing RAG ")
    # Aquí se puede implementar la lógica para generar una respuesta utilizando RAG
    nodes = service.rag_service.retrieve_data(prompt)
    response = service.llm_slow.response_with_data(prompt, nodes)
    return f"RAG response generated for prompt: {prompt}\nResponse: {response}"
