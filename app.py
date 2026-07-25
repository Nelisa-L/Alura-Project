"""
app.py
------
Interfaz web (Streamlit) para interactuar con el agente.

Detecta automaticamente el entorno de ejecucion:
- Si existe la variable de entorno GROQ_API_KEY -> usa Groq (pensado para
  despliegue en la nube, por ejemplo Render, donde no es viable correr
  Ollama por falta de RAM).
- Si no existe -> usa Ollama (pensado para desarrollo local).

Ejecutar con:
    streamlit run app.py
"""
import os
import streamlit as st

USE_CLOUD = bool(os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="Asistente NelisaPay", page_icon="💬")
st.title("💬 Asistente Virtual — NelisaPay")
st.caption(
    "Agente de IA que responde preguntas sobre privacidad, terminos de uso, "
    "limites de transacciones, seguridad y tarifas, basandose en la "
    "documentacion oficial cargada en /data."
)

if USE_CLOUD:
    st.caption("🌐 Modo nube activo (motor: Groq)")
else:
    st.caption("💻 Modo local activo (motor: Ollama)")


@st.cache_resource(show_spinner="Cargando el agente y el indice de documentos...")
def load_agent():
    if USE_CLOUD:
        from src.agent_cloud import DocumentAgentCloud
        return DocumentAgentCloud()
    else:
        from src.agent import DocumentAgent
        return DocumentAgent()


try:
    agent = load_agent()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy el asistente de NelisaPay. "
                                          "Puedes preguntarme sobre limites de transacciones, "
                                          "tarifas, seguridad o privacidad."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            answer, sources = agent.ask_with_sources(question)
        st.write(answer)
        with st.expander("Ver fuentes consultadas"):
            for s in sources:
                st.markdown(f"**{s['source']}**")
                st.caption(s["preview"] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
