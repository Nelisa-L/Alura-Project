"""
app.py
------
Interfaz web (Streamlit) para interactuar con el agente de forma
conversacional. Ejecutar con:

    streamlit run app.py
"""
import streamlit as st
from src.agent import DocumentAgent

st.set_page_config(page_title="Asistente NelisaPay", page_icon="💬")
st.title("💬 Asistente Virtual — NelisaPay")
st.caption(
    "Agente de IA que responde preguntas sobre privacidad, términos de uso, "
    "límites de transacciones, seguridad y tarifas, basándose en la "
    "documentación oficial cargada en /data."
)


@st.cache_resource(show_spinner="Cargando el agente y el índice de documentos...")
def load_agent():
    return DocumentAgent()


try:
    agent = load_agent()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy el asistente de NelisaPay. "
                                          "Puedes preguntarme sobre límites de transacciones, "
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
