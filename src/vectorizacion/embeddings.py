from archivo_barril import *

# Configuración idéntica a la ingesta
MONGO_URI       = "mongodb://localhost:27017"
DB_NAME         = "pruebas"
COLLECTION_NAME = "documentos_chunk"

client     = MongoClient(MONGO_URI)
db         = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Se carga el modelo all-MiniLM-L6-v2
model = SentenceTransformer("all-MiniLM-L6-v2")

def vectorizar_chunks(batch_size: int = 32):
    """
    1) Busca documentos sin vector.
    2) Lee su texto_chunk.
    3) Calcula embeddings en batches.
    4) Actualiza los documentos con 'vector' y 'fecha_vectorizacion'.
    """
    docs = list(collection.find({"vector": {"$exists": False}}))
    if not docs:
        print("No hay chunks nuevos para vectorizar.")
        return

    textos = [d["texto_chunk"] for d in docs]
    ids     = [d["_id"]          for d in docs]
    ahora = datetime.now(timezone.utc)

    total = len(textos)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = textos[start:end]
        batch_ids   = ids[start:end]

        # 1) Generar embeddings
        vects = model.encode(batch_texts, show_progress_bar=True)

        # 2) Actualizar en Mongo
        for doc_id, vec in zip(batch_ids, vects):
            collection.update_one(
                {"_id": doc_id},
                {"$set": {
                    "vector": vec.tolist(),
                    "fecha_vectorizacion": ahora
                }}
            )
        print(f"Vectorizados documentos {start+1}-{end} de {total}")

if __name__ == "__main__":
    vectorizar_chunks(batch_size=32)
