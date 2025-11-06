import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
import requests
from huggingface_hub import InferenceClient

# --------------------------------------------------------
# CONFIGURACIÓN INICIAL
# --------------------------------------------------------
st.set_page_config(page_title="Taller IA: OCR + LLM")
st.title("Taller IA: OCR + LLM")
st.write("Aplicación multimodal: OCR + Análisis de texto con GROQ y Hugging Face")

# --------------------------------------------------------
# CARGAR CLAVES DESDE .env
# --------------------------------------------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
hf_key = os.getenv("HUGGINGFACE_API_KEY")

if not groq_key or not hf_key:
    st.error("⚠️ Faltan claves de API. Verifica tu archivo .env.")
    st.stop()

# --------------------------------------------------------
# CARGAR EL MODELO OCR SOLO UNA VEZ
# --------------------------------------------------------
@st.cache_resource
def cargar_lector():
    return easyocr.Reader(['es', 'en'])

reader = cargar_lector()

# --------------------------------------------------------
# MÓDULO 1: OCR
# --------------------------------------------------------
st.header("Módulo 1: Lector de Imágenes (OCR)")

uploaded_file = st.file_uploader("Sube una imagen (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])

# Inicializar texto en session_state
if "texto_extraido" not in st.session_state:
    st.session_state["texto_extraido"] = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Extraer texto"):
        img_array = np.array(image)
        with st.spinner("Extrayendo texto..."):
            resultado = reader.readtext(img_array, detail=0)
            st.session_state["texto_extraido"] = "\n".join(resultado)
        st.success("✅ Texto extraído y guardado.")

# Mostrar texto persistente
if st.session_state["texto_extraido"]:
    st.text_area("Texto extraído:", st.session_state["texto_extraido"], height=200)
else:
    st.info("Aún no se ha extraído texto.")


# --------------------------------------------------------
# MÓDULO 2: Conexión con el Cerebro Lingüístico (GROQ API)
# --------------------------------------------------------
st.header("Módulo 2: Conexión con el Cerebro Lingüístico (GROQ API)")

# Selección de modelo y tarea
modelo_groq = st.selectbox(
    "Selecciona el modelo de GROQ:",
    ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768-v0.1"]
)

tarea = st.selectbox(
    "Selecciona la tarea a realizar sobre el texto:",
    ["Resumir en 3 puntos clave", "Identificar las entidades principales", "Traducir al inglés"]
)

# Botón para ejecutar análisis
if st.button("Analizar texto con GROQ (Módulo 2)"):
    texto = st.session_state["texto_extraido"].strip()

    if not texto:
        st.warning("Primero debes extraer texto de una imagen.")
    else:
        st.info(f"Procesando con modelo **{modelo_groq}**...")

        # Construir el mensaje para el modelo
        messages = [
            {"role": "system", "content": f"Eres un asistente de procesamiento de texto. Tu tarea es: {tarea}."},
            {"role": "user", "content": f"A continuación tienes el texto a analizar:\n\n{texto}"}
        ]

        # Crear la solicitud a la API de GROQ
        payload = {
            "model": modelo_groq,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512
        }

        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }

        endpoint = "https://api.groq.com/openai/v1/chat/completions"

        try:
            with st.spinner("Obteniendo respuesta del modelo..."):
                response = requests.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                contenido = data["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error al conectar con GROQ: {e}")
        else:
            st.markdown("### Respuesta del modelo (GROQ):")
            st.markdown(contenido)

# --------------------------------------------------------
# MÓDULO 3: Flexibilidad y Experimentación
# --------------------------------------------------------
st.header("Módulo 3: Análisis con LLMs (GROQ / Hugging Face)")

# Elegir proveedor
proveedor = st.radio("Selecciona el proveedor de modelo:", ["GROQ", "Hugging Face"])

# Sliders para parámetros
temperature = st.slider("Temperature (creatividad):", 0.0, 1.0, 0.3, 0.1)
max_tokens = st.slider("Máximo de tokens (longitud de respuesta):", 100, 1000, 512, 50)

# Selección de modelo y tarea (comunes)
modelo_groq = "llama-3.3-70b-versatile"
modelo_hf = "facebook/bart-large-cnn"  # modelo de resumen por defecto

tarea = st.selectbox(
    "Selecciona la tarea a realizar:",
    ["Resumir en 3 puntos clave", "Identificar las entidades principales", "Traducir al inglés"]
)

# --------------------------------------------------------
# BOTÓN PARA ANALIZAR TEXTO
# --------------------------------------------------------
if st.button("Analizar Texto"):
    texto = st.session_state["texto_extraido"].strip()

    if not texto:
        st.warning("⚠️ Primero debes extraer texto de una imagen.")
        st.stop()

    if proveedor == "GROQ":
        # --------------------- GROQ ---------------------
        st.info(f"Procesando con **GROQ** ({modelo_groq})...")

        messages = [
            {"role": "system", "content": f"Eres un asistente de texto. Tu tarea es: {tarea}."},
            {"role": "user", "content": f"Texto a analizar:\n\n{texto}"}
        ]

        payload = {
            "model": modelo_groq,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }

        endpoint = "https://api.groq.com/openai/v1/chat/completions"

        try:
            with st.spinner("Conectando con GROQ..."):
                response = requests.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                contenido = data["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error al conectar con GROQ: {e}")
        else:
            st.markdown("### Respuesta del modelo (GROQ):")
            st.markdown(contenido)

elif proveedor == "Hugging Face":
    st.info(f"Procesando con **Hugging Face**...")
    try:
        client = InferenceClient(api_key=hf_key)
        with st.spinner("Conectando con Hugging Face..."):
            if "Resumir" in tarea:
                output = client.summarization(
                    model="facebook/bart-large-cnn",
                    inputs=texto
                )
                resultado = output[0]["summary_text"]

            elif "Traducir" in tarea:
                output = client.translation(
                    model="Helsinki-NLP/opus-mt-es-en",
                    inputs=texto
                )
                resultado = output[0]["translation_text"]

            else:
                output = client.text_generation(
                    model="tiiuae/falcon-7b-instruct",
                    inputs=f"Identifica las entidades principales en el siguiente texto:\n\n{texto}",
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                resultado = output
    except Exception as e:
        st.error(f"❌ Error al conectar con Hugging Face: {e}")
    else:
        st.markdown("### 🤖 Respuesta del modelo (Hugging Face):")
        st.markdown(resultado)
