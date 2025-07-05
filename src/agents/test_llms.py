from src.agents.llm import LLMConfig, LLMService, RAGConfig, RAGService
from src.agents.pipelines import (
    AIService,
    pipeline_documentation,
    pipeline_extended_text,
    pipeline_summary,
)
from src.constants import LOGGER


def test_completion():
    questions = [
        "con que sitios colaboramos?",
        "SDP Technology Architecture, whats next?",
        "What links there are between the documents?",
    ]
    # ejemplo con llama
    llm_service = LLMConfig().get_agent_model()
    for question in questions:
        response = llm_service.complete(question)
        LOGGER.info(f"Pregunta: {question}")
        LOGGER.info(f"Respuesta: {response}")

        assert not response.text.endswith("Empty response") and len(response.text) != 0, (
            "La respuesta del LLM no debe ser vacía."
        )


# ---------------------------------------------------------------------------


def test_pipelines():
    questions = [
        "quisiera reducir el contenido del siguiente texto 'La arquitectura tecnológica de SDP es una estructura compleja que integra múltiples componentes y servicios para ofrecer soluciones eficientes y escalables.'",
        "necesito hacer un documento de arquitectura tecnológica para SDP",
        "necesito escribir un correo que en diga 'el proyecto SDP ha salido adelante y sigue todas las previsiones', alargalo y añade descripciones para que sea un texto extenso completo",
    ]

    ai_service = AIService(
        llm_slow=LLMService.from_config(LLMConfig()),
        llm_fast=LLMService.from_config(
            LLMConfig()  # llm_model_name="meta-llama/Llama-3.2-1B-Instruct")
        ),
        rag_service=RAGService.from_config(RAGConfig()),
        pipelines=[
            pipeline_documentation(),
            pipeline_summary(),
            pipeline_extended_text(),
        ],  # Aquí podrías añadir pipelines si los tienes definidos
    )
    for question in questions:
        LOGGER.info("_" * 80)
        LOGGER.info(f"Pregunta: {question}")
        pipeline = ai_service.select_pipeline(question)

        LOGGER.info(f"Pipeline seleccionados: {[p.name for p in pipeline]}")
        LOGGER.info("_" * 80)


# ---------------------------------------------------------------------------
