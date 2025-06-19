from archivo_barril import *

from constants import DB_NAME, MONGO_COLLECTION, MONGO_HOST

client = MongoClient(MONGO_HOST)  # client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
collection = db[MONGO_COLLECTION]


class LectorPDF_Local:
    def __init__(self, carpeta_pdfs: str):
        """
        Inicializa el lector local con la ruta a la carpeta donde están
        los archivos .pdf.
        """
        self.carpeta = carpeta_pdfs

    def listar_fuentes(self) -> List[str]:
        """
        Devuelve un listado de rutas completas a todos los .pdf
        dentro de la carpeta configurada.
        """
        archivos_pdf = []
        for nombre in os.listdir(self.carpeta):
            if nombre.lower().endswith(".pdf"):
                ruta_completa = os.path.join(self.carpeta, nombre)
                archivos_pdf.append(ruta_completa)
        return archivos_pdf

    def abrir_fuente(self, fuente: str) -> bytes:
        """
        Dado el path completo a un PDF local, devuelve los bytes del archivo.
        """
        with open(fuente, "rb") as f:
            return f.read()


class ExtractorTextoPDF:
    def extraer_texto(self, pdf_bytes: bytes) -> str:
        """
        Recibe los bytes de un PDF y devuelve todo el texto concatenado
        página a página usando PyMuPDF.
        """
        texto_total = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for pagina in doc:
                # .get_text() extrae todo el texto de la página
                texto_pagina = pagina.get_text() or ""
                texto_total += texto_pagina + "\n"
        return texto_total


def ingesta_pdfs_local(carpeta_pdfs: str) -> None:
    """
    Orquesta el flujo completo de ingesta de PDFs locales y guarda cada chunk
    en MongoDB:
      1) Lista todos los .pdf en la carpeta.
      2) Para cada PDF, extrae el texto completo con PyMuPDF.
      3) Limpia y segmenta el texto en chunks (por palabras).
      4) Inserta cada chunk en MongoDB con metadatos (sin vectorización).

    Parámetros:
    - carpeta_pdfs: ruta al directorio donde están los PDFs.
    """
    lector = LectorPDF_Local(carpeta_pdfs)
    extractor = ExtractorTextoPDF()
    ahora = datetime.utcnow()

    for ruta_pdf in lector.listar_fuentes():
        # 1. Abrir y leer PDF
        pdf_bytes = lector.abrir_fuente(ruta_pdf)

        # 2. Extraer texto plano con PyMuPDF
        texto_completo = extractor.extraer_texto(pdf_bytes)

        # 3. Limpieza básica (eliminar líneas vacías repetidas)
        lineas = [linea.strip() for linea in texto_completo.splitlines()]
        texto_limpio = "\n".join([linea for linea in lineas if linea])

        # 4. Segmentar en chunks basados en palabras
        lista_chunks = segmentar_texto(texto_limpio, max_palabras_por_chunk=200, solapamiento=50)

        nombre_archivo = os.path.basename(ruta_pdf)

        # 5. Construir documentos e insertar en MongoDB
        documentos_a_insertar = []
        for i, chunk in enumerate(lista_chunks):
            doc = {
                "document_id": f"{nombre_archivo}_chunk_{i}",
                "chunk_index": i,
                "nombre_archivo": nombre_archivo,
                "ruta": ruta_pdf,
                "texto_chunk": chunk,
                "fecha_ingesta": ahora,
            }
            documentos_a_insertar.append(doc)

        if documentos_a_insertar:
            collection.insert_many(documentos_a_insertar)
            print(f"Insertados {len(documentos_a_insertar)} chunks de: {nombre_archivo}")


# Bloque de prueba: Descomentar para probar cosas.
# if __name__ == "__main__":
#    carpeta = r"C:\Users\carlo\Documents\GitHub\MIOTI_DL_ProyectoFinal\Document\standards_management_and_harmonization"
#    ingesta_pdfs_local(carpeta)

# 1) Mostrar cuántos chunks hay en total
# total_chunks = collection.count_documents({})
# print(f"Total de chunks almacenados: {total_chunks}\n")

# 2) Listar los primeros 5 chunks (solo metadatos breves)
# print("=== Primeros 5 chunks (document_id, nombre_archivo, chunk_index) ===")
# for doc in collection.find().limit(5):
#    print(f"- {doc['document_id']} | {doc['nombre_archivo']} | índice {doc['chunk_index']}")
