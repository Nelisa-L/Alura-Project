"""
agent_cloud.py
---------------
Version del motor RAG pensada para DESPLIEGUE EN LA NUBE (Render, etc.),
donde no es viable correr Ollama ni cargar modelos de embeddings en RAM
local (el tier gratuito de Render solo ofrece 512MB).

En su lugar, usa:
- Groq (API gratuita) para la generacion de respuestas (LLM).
- API de inferencia de HuggingFace (llamada remota via HTTP) para los
  embeddings. No carga ningun modelo pesado en memoria local.

Requiere las variables de entorno GROQ_API_KEY y HF_TOKEN configuradas
(nunca las escribas directamente en el codigo).
"""
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
HF_API_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modelo servido por Groq. Llama 3.1 8B es rapido y gratuito en su tier free.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """Eres el asistente virtual de soporte de NelisaPay, una \
plataforma de pagos digitales. Respondes preguntas de los usuarios UNICAMENTE \
usando la informacion del CONTEXTO proporcionado, que proviene de la \
documentacion oficial (politica de privacidad, terminos y condiciones, \
preguntas frecuentes, politica de seguridad y tarifas).

Reglas:
- Si la respuesta esta en el contexto, respondela de forma clara, breve y directa.
- Si la informacion NO esta en el contexto, di explicitamente que no cuentas \
con esa informacion en la documentacion disponible y sugiere contactar a soporte.
- No inventes cifras, plazos ni politicas que no esten en el contexto.
- Responde siempre en espanol, con un tono profesional y cercano.

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
                "No se encontro el indice vectorial. Ejecuta primero: "
                "python src/ingest.py"
            )

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "Falta la variable de entorno GROQ_API_KEY. "
                "Configurala con tu API key de https://console.groq.com"
            )

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Falta la variable de entorno HF_TOKEN. "
                "Crea un token gratuito (read) en https://huggingface.co/settings/tokens"
            )

        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name=HF_API_EMBEDDING_MODEL,
        )
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
        q = input("Tu: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        respuesta = agent.ask(q)
        print(f"\nAgente: {respuesta}\n")
