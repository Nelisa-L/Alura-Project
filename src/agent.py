"""
agent.py
--------
Motor del agente de preguntas y respuestas (RAG - Retrieval-Augmented
Generation). Carga el índice vectorial ya construido por ingest.py,
busca los fragmentos más relevantes para la pregunta del usuario y
se los pasa como contexto a un modelo de lenguaje local servido por
Ollama, para que genere una respuesta basada únicamente en esa
información.

Requiere que Ollama esté instalado y corriendo (`ollama serve`) y que
el modelo indicado (OLLAMA_MODEL) haya sido descargado previamente
con `ollama pull <modelo>`.
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Puedes cambiar el modelo por cualquier otro que hayas descargado con Ollama,
# por ejemplo: "mistral", "phi3", "llama3.1:8b", "qwen2.5:7b"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

SYSTEM_PROMPT = """Eres el asistente virtual de soporte de NelisaPay, una \
plataforma de pagos digitales. Respondes preguntas de los usuarios ÚNICAMENTE \
usando la información del CONTEXTO proporcionado, que proviene de la \
documentación oficial (política de privacidad, términos y condiciones, \
preguntas frecuentes, política de seguridad y tarifas).

Reglas:
- Si la respuesta está en el contexto, respóndela de forma clara, breve y directa.
- Si la información NO está en el contexto, di explícitamente que no cuentas \
con esa información en la documentación disponible y sugiere contactar a soporte.
- No inventes cifras, plazos ni políticas que no estén en el contexto.
- Responde siempre en español, con un tono profesional y cercano.

Contexto:
{context}
"""


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"(Fuente: {d.metadata.get('source', 'desconocida')})\n{d.page_content}"
        for d in docs
    )


class DocumentAgent:
    def __init__(self, k: int = 4):
        if not os.path.exists(VECTORSTORE_DIR):
            raise RuntimeError(
                "No se encontró el índice vectorial. Ejecuta primero: "
                "python src/ingest.py"
            )

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.chain = (
            {"context": self.retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str):
        """Devuelve la respuesta junto con los fragmentos fuente usados,
        útil para depurar o mostrar transparencia al usuario."""
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        sources = [
            {"source": d.metadata.get("source", "desconocida"),
             "preview": d.page_content[:150]}
            for d in docs
        ]
        return answer, sources


if __name__ == "__main__":
    agent = DocumentAgent()
    print("Agente listo. Escribe 'salir' para terminar.\n")
    while True:
        q = input("Tú: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        respuesta = agent.ask(q)
        print(f"\nAgente: {respuesta}\n")
