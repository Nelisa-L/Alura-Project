# NelisaPay — Agente de IA para Documentación de Servicios Financieros

## 📋 Descripción general

Este proyecto implementa un **agente de inteligencia artificial conversacional**
capaz de responder preguntas en lenguaje natural basándose en documentos reales
de una fintech: política de privacidad, términos y condiciones, preguntas
frecuentes sobre transacciones y límites, política de seguridad y prevención
de fraudes, y tarifas del servicio.

El agente utiliza la técnica **RAG (Retrieval-Augmented Generation)**: en
lugar de que el modelo de lenguaje "invente" respuestas, primero se buscan
los fragmentos más relevantes dentro de los documentos fuente (PDF y CSV),
y luego se le pide al modelo que redacte la respuesta **basándose únicamente
en esa información**. Esto reduce alucinaciones y asegura que las respuestas
reflejen la documentación oficial vigente.

Todo el proyecto corre **100% de forma local**, sin necesidad de API keys de
pago ni conexión a servicios en la nube:

- Los **embeddings** (representación numérica del texto) se generan con un
  modelo local de **HuggingFace** (`sentence-transformers`).
- El **modelo de lenguaje** que redacta las respuestas corre localmente con
  **Ollama**.

## 🏗️ Arquitectura de la solución

```
                         ┌─────────────────────────┐
                         │   Documentos fuente      │
                         │  (PDF y CSV en /data)     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   src/ingest.py          │
                          │  - Carga PDF/CSV         │
                          │  - Divide en fragmentos  │
                          │    (chunks)              │
                          │  - Genera embeddings     │
                          │    (HuggingFace)         │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  vectorstore/ (FAISS)    │
                          │  Índice vectorial local  │
                          └────────────┬─────────────┘
                                       │
              Pregunta del usuario     │
                     │                 │
                     ▼                 ▼
             ┌──────────────────────────────────┐
             │          src/agent.py              │
             │  1. Busca los fragmentos más       │
             │     relevantes (retriever)         │
             │  2. Arma un prompt con el contexto │
             │     encontrado                     │
             │  3. Envía el prompt al modelo LLM  │
             │     (Ollama, local)                │
             │  4. Devuelve la respuesta generada │
             └───────────────┬────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   app.py (Streamlit)     │
                 │   Interfaz de chat web    │
                 └─────────────────────────┘
```

**Flujo resumido:**
1. `ingest.py` procesa los documentos una sola vez y construye el índice vectorial.
2. `agent.py` carga ese índice y responde preguntas combinando búsqueda semántica + generación con LLM.
3. `app.py` expone todo esto en una interfaz de chat web amigable.

## 🛠️ Tecnologías y herramientas utilizadas

| Componente | Herramienta | Rol |
|---|---|---|
| Lenguaje | Python 3.10+ | Lenguaje principal del proyecto |
| Orquestación RAG | LangChain | Conecta retriever + prompt + LLM |
| Embeddings | HuggingFace `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | Convierte texto en vectores, modelo local y gratuito, multilingüe |
| Índice vectorial | FAISS | Búsqueda semántica rápida de fragmentos relevantes |
| Modelo de lenguaje (LLM) | Ollama (`llama3.1:8b` por defecto) | Genera las respuestas en lenguaje natural, corre 100% local |
| Lectura de PDF | `pypdf` / `PyPDFLoader` | Extrae texto de la política de privacidad, T&C, etc. |
| Lectura de CSV | `pandas` | Extrae preguntas frecuentes y tarifas tabuladas |
| Interfaz | Streamlit | Chat web para interactuar con el agente |
| Editor | VS Code | Entorno de desarrollo |

## ▶️ Instrucciones para ejecutar el proyecto

### 1. Requisitos previos
- Python 3.10 o superior
- [Ollama](https://ollama.com/download) instalado en tu computadora

### 2. Instalar Ollama y descargar el modelo
```bash
# Instala Ollama (ver instrucciones según tu sistema operativo en ollama.com)
# Luego descarga el modelo que usará el agente:
ollama pull llama3.1:8b
```

### 3. Clonar el repositorio y crear el entorno virtual
```bash
git clone https://github.com/Nelisa-L/Alura-Project.git
cd Alura-Project
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Agregar tus documentos
Coloca tus archivos PDF y/o CSV dentro de la carpeta `/data`.
El repositorio ya incluye documentos de ejemplo (`politica_privacidad.pdf`
y `faq_tarifas.csv`) para que puedas probar el agente de inmediato.

### 5. Construir el índice vectorial (solo la primera vez o al actualizar documentos)
```bash
python src/ingest.py
```

### 6. Ejecutar el agente

**Opción A — Interfaz de chat web (recomendada):**
```bash
streamlit run app.py
```
Se abrirá automáticamente en tu navegador (normalmente `http://localhost:8501`).

**Opción B — Terminal:**
```bash
python src/agent.py
```

## ❓ Ejemplos de preguntas que el agente puede responder

- ¿Cuál es el límite diario de transferencia para una cuenta verificada?
- ¿Cuánto cuesta transferir dinero a una cuenta bancaria externa?
- ¿NelisaPay comparte mis datos personales con terceros?
- ¿Qué hago si sospecho que recibí un mensaje de phishing?
- ¿Cuánto tarda una transferencia a un banco externo?
- ¿Cómo solicito la eliminación de mi cuenta y mis datos?
- ¿Cuándo se requiere autenticación de dos factores?
- ¿Hay comisión por mantener mi cuenta inactiva?

## 💬 Ejemplos de respuestas generadas por el agente

*Capturas reales obtenidas ejecutando el agente localmente (Ollama + `llama3.1:8b`,
embeddings con HuggingFace, interfaz en Streamlit).*

> **Pregunta:** ¿Cuál es el límite diario de transferencia para una cuenta verificada?
>
> **Agente:** Según la información disponible, el límite diario de transferencia
> para una cuenta verificada es de $5,000 USD.

> **Pregunta:** ¿Cuánto cuesta transferir a una cuenta bancaria externa?
>
> **Agente:** Según la política de privacidad, las transferencias a cuentas bancarias
> externas tienen una comisión fija de $1.50 USD por transacción.

> **Pregunta:** ¿NelisaPay comparte mis datos con terceros?
>
> **Agente:** Según nuestra política de privacidad, no. NelisaPay no vende ni comparte
> datos personales con terceros con fines comerciales.

> **Pregunta:** ¿Puedo invertir en criptomonedas con NelisaPay?
>
> **Agente:** Lo siento, pero no tengo información sobre la posibilidad de invertir en
> criptomonedas con NelisaPay en el contexto proporcionado. La documentación disponible
> solo menciona cambio de divisas dentro de la plataforma.

*(Este último ejemplo muestra cómo el agente reconoce cuándo una pregunta está
fuera del alcance de los documentos cargados, evitando inventar una respuesta.)*

## 📁 Estructura del repositorio

```
Alura-Project/
├── data/                       # Documentos fuente (PDF y CSV)
│   ├── politica_privacidad.pdf
│   ├── faq_tarifas.csv
│   └── generate_sample_pdf.py  # Script que generó el PDF de ejemplo
├── src/
│   ├── ingest.py                # Procesa documentos y construye el índice vectorial
│   └── agent.py                 # Lógica del agente (RAG + Ollama)
├── vectorstore/                 # Índice FAISS generado (no se versiona en git)
├── app.py                       # Interfaz de chat en Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Despliegue

Este proyecto puede desplegarse en un servidor propio (por ejemplo, una
instancia gratuita de Oracle Cloud Infrastructure - OCI Ampere A1),
instalando Ollama en el servidor y ejecutando `streamlit run app.py`
detrás de un proxy con HTTPS.

## 📌 Notas

- Los documentos incluidos en `/data` son **ficticios**, creados únicamente
  con fines de demostración. Reemplázalos por tus documentos reales.
- Puedes cambiar el modelo de Ollama utilizado editando la variable
  `OLLAMA_MODEL` en `src/agent.py` o mediante la variable de entorno del
  mismo nombre.
