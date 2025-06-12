from archivo_barril import *

def segmentar_texto(
    texto: str,
    max_palabras_por_chunk: int = 200,
    solapamiento: int = 50
) -> List[str]:
    """
    Divide el texto en fragmentos (“chunks”) de aproximadamente
    max_palabras_por_chunk palabras, con solapamiento de solapamiento
    palabras entre chunks consecutivos.

    - texto: texto completo a segmentar.
    - max_palabras_por_chunk: número máximo de palabras por cada chunk.
    - solapamiento: cuántas palabras se repiten entre el chunk n y el n+1.

    Devuelve una lista de strings, cada uno con el texto de un chunk.
    """
    palabras = texto.split()
    if not palabras:
        return []

    chunks = []
    inicio = 0
    total = len(palabras)

    while inicio < total:
        fin = inicio + max_palabras_por_chunk
        if fin >= total:
            fin = total
            # Cortamos aquí porque ya alcanzamos el final
            chunk = " ".join(palabras[inicio:fin])
            chunks.append(chunk)
            break

        chunk = " ".join(palabras[inicio:fin])
        chunks.append(chunk)

        # Avanzamos teniendo en cuenta el solapamiento
        inicio = fin - solapamiento
        # Si por algún motivo se calculase negativo, lo dejamos en cero
        if inicio < 0:
            inicio = 0

    return chunks
