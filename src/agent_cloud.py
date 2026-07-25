"""
agent_cloud.py
---------------
Version del motor RAG pensada para DESPLIEGUE EN LA NUBE (Render, etc.),
donde no es viable correr Ollama ni sentence-transformers con PyTorch
(requieren mas RAM de la que ofrecen los tiers gratuitos, 512MB).

En su lugar, usa:
- Groq (API gratuita) para la generacion de respuestas (LLM).
- FastEmbed (ONNX, sin PyTorch) para los embeddings, mucho mas liviano
  en memoria que sentence-transformers.

Requiere la variable de entorno GROQ_API_KEY configurada (nunca la
escribas directamente en el codigo).
"""
import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modelo servido por Groq. Llama 3.1 8B es rápido y gratuito en su tier free.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

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


class DocumentAgentCloud:
    def __init__(self, k: int = 4):
        if not os.path.exists(VECTORSTORE_DIR):
            raise RuntimeError(
                "No se encontró el índice vectorial. Ejecuta primero: "
                "python src/ingest.py"
            )

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "Falta la variable de entorno GROQ_API_KEY. "
                "Configúrala con tu API key de https://console.groq.com"
            )

        embeddings = FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=groq_key)

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
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        sources = [
            {"source": d.metadata.get("source", "desconocida"),
             "preview": d.page_content[:150]}
            for d in docs
        ]
        return answer, sources


if __name__ == "__main__":
    agent = DocumentAgentCloud()
    print("Agente (Groq) listo. Escribe 'salir' para terminar.\n")
    while True:
        q = input("Tú: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        respuesta = agent.ask(q)
        print(f"\nAgente: {respuesta}\n")
