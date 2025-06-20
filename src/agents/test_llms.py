from src.agents.llm import LLMConfig


def test_completion():
    # ejemplo con llama
    llm_service = LLMConfig().get_agent_model()
    questions = [
        "con que sitios colaboramos?",
        "SDP Technology Architecture, whats next?",
        "What links there are between the documents?",
    ]
    for question in questions:
        response = llm_service.complete(question)
        print(f"Pregunta: {question}")
        print(f"Respuesta: {response}")

        assert not response.text.endswith("Empty response") and len(response.text) != 0, (
            "La respuesta del LLM no debe ser vacía."
        )


# ---------------------------------------------------------------------------
