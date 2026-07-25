"""
ingest.py
---------
Lee todos los documentos soportados (PDF y CSV) dentro de /data,
los divide en fragmentos (chunks), genera sus embeddings y construye
un indice vectorial FAISS que se guarda en /vectorstore.

Este script se ejecuta una sola vez (o cada vez que cambian los
documentos fuente). El agente (agent.py / agent_cloud.py) luego solo
carga el indice ya construido, sin necesidad de reprocesar los documentos.

Motor de embeddings segun el entorno:
- Local (desarrollo, sin GROQ_API_KEY): sentence-transformers (HuggingFace),
  mas preciso pero requiere PyTorch (~500MB+ de RAM).
- Nube (deploy, con GROQ_API_KEY definido): FastEmbed (ONNX, sin PyTorch),
  mucho mas liviano en RAM, apto para el tier gratuito de Render (512MB).
"""
import os
import glob
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

USE_CLOUD = bool(os.environ.get("GROQ_API_KEY"))

# Modelo local (sentence-transformers), multilingue, usado en desarrollo
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modelo liviano (FastEmbed/ONNX), multilingue, usado en la nube (bajo RAM)
FASTEMBED_MODEL = "intfloat/multilingual-e5-small"


def get_embeddings():
    """Devuelve el motor de embeddings segun el entorno de ejecucion."""
    if USE_CLOUD:
        from langchain_community.embeddings import FastEmbedEmbeddings
        print(f"[ingest] Usando FastEmbed (modo nube, bajo consumo de RAM): {FASTEMBED_MODEL}")
        return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"[ingest] Usando HuggingFace sentence-transformers (modo local): {HF_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


def load_pdfs() -> list[Document]:
    """Carga todos los PDF de /data como documentos de LangChain."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(path)
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = os.path.basename(path)
        docs.extend(pages)
        print(f"[ingest] PDF cargado: {os.path.basename(path)} ({len(pages)} paginas)")
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[ingest] Total de fragmentos generados: {len(chunks)}")

    embeddings = get_embeddings()
    print("[ingest] Generando embeddings (puede tardar la primera vez)...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"[ingest] Indice vectorial guardado en: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
