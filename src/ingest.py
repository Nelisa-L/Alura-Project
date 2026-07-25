"""
ingest.py
---------
Lee todos los documentos soportados (PDF y CSV) dentro de /data,
los divide en fragmentos (chunks), genera sus embeddings con un
modelo local de HuggingFace (sentence-transformers) y construye
un índice vectorial FAISS que se guarda en /vectorstore.

Este script se ejecuta una sola vez (o cada vez que cambian los
documentos fuente). El agente (agent.py) luego solo carga el índice
ya construido, sin necesidad de reprocesar los documentos.
"""
import os
import glob
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# Modelo de embeddings local, gratuito, liviano y multilingüe
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_pdfs() -> list[Document]:
    """Carga todos los PDF de /data como documentos de LangChain."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(path)
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = os.path.basename(path)
        docs.extend(pages)
        print(f"[ingest] PDF cargado: {os.path.basename(path)} ({len(pages)} páginas)")
    return docs


def load_csvs() -> list[Document]:
    """Carga todos los CSV de /data. Cada fila se convierte en un documento,
    combinando sus columnas en un texto legible para el modelo."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        df = pd.read_csv(path)
        for i, row in df.iterrows():
            content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            docs.append(Document(
                page_content=content,
                metadata={"source": os.path.basename(path), "row": i}
            ))
        print(f"[ingest] CSV cargado: {os.path.basename(path)} ({len(df)} filas)")
    return docs


def build_vectorstore():
    all_docs = load_pdfs() + load_csvs()

    if not all_docs:
        raise RuntimeError(
            f"No se encontraron documentos PDF o CSV en {DATA_DIR}. "
            "Agrega tus archivos y vuelve a ejecutar este script."
        )

    # Dividimos los documentos largos (como el PDF) en fragmentos manejables.
    # Los documentos de CSV ya son cortos (una fila = una idea), así que
    # el splitter los deja prácticamente intactos.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[ingest] Total de fragmentos generados: {len(chunks)}")

    print(f"[ingest] Generando embeddings con: {EMBEDDING_MODEL} (puede tardar la primera vez)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"[ingest] Índice vectorial guardado en: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
