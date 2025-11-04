import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
import requests

st.set_page_config(page_title="Taller IA: OCR + LLM")

# Título principal
st.title("Taller IA: OCR + LLM")

# Encabezado de sección
st.header("Módulo 1: Lector de Imágenes (OCR)")

@st.cache_resource
def cargar_lector():
    lector = easyocr.Reader(['es', 'en'])  # idiomas: español e inglés
    return lector

reader = cargar_lector()  # Se carga una sola vez


uploaded_file = st.file_uploader("Sube una imagen (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Convertir a formato que EasyOCR pueda procesar
    img_array = np.array(image)

    # --- Ejecutar OCR ---
    with st.spinner("Extrayendo texto..."):
        resultado = reader.readtext(img_array, detail=0)

    # Mostrar resultado
    texto_extraido = "\n".join(resultado)
    st.text_area("Texto extraído:", texto_extraido, height=200)

# --- Configuración de la página ---
st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon="🤖")
st.title("Taller IA: OCR + LLM")
st.header("Módulo 2: Conexión con el Cerebro Lingüístico (GROQ API)")

# --- Cargar claves desde .env ---
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# Verificar clave
if not groq_key:
    st.error("No se encontró la clave GROQ_API_KEY en el archivo .env.")
    st.stop()

# Supongamos que ya tienes el texto extraído en una variable llamada `texto_extraido`
# (por ejemplo: resultado del OCR)

# Parámetros de la UI (selectbox ya definidos en pasos anteriores):
modelo_groq = st.selectbox(
    "Selecciona el modelo de GROQ:",
    ["llama3-8b-8192", "mixtral-8x7b-32768"]
)

tarea = st.selectbox(
    "Selecciona la tarea a realizar sobre el texto:",
    ["Resumir en 3 puntos clave", "Identificar las entidades principales", "Traducir al inglés"]
)

st.header("Módulo 3: Análisis con LLMs (GROQ / Hugging Face)")

# Elegir proveedor
proveedor = st.radio("Selecciona el proveedor de modelo:", ["GROQ", "Hugging Face"])

# Sliders para parámetros
temperature = st.slider("Temperature (creatividad):", 0.0, 1.0, 0.3, 0.1)
max_tokens = st.slider("Máximo de tokens (longitud de respuesta):", 100, 1000, 512, 50)

# Selección de modelo y tarea (comunes)
modelo_groq = "llama3-8b-8192"
modelo_hf = "facebook/bart-large-cnn"  # modelo de resumen por defecto

tarea = st.selectbox(
    "Selecciona la tarea a realizar:",
    ["Resumir en 3 puntos clave", "Identificar las entidades principales", "Traducir al inglés"]
)


if st.button("Analizar Texto"):
    if not st.session_state["texto_extraido"].strip():
        st.warning("Primero debes extraer texto de una imagen.")
    else:
        st.info(f"Analizando texto con el modelo **{modelo_groq}**...")
        texto = st.session_state["texto_extraido"]

        # Crear mensajes para el modelo
        messages = [
            {"role": "system", "content": f"Eres un asistente de procesamiento de texto. Tu tarea es: {tarea}."},
            {"role": "user", "content": f"A continuación tienes el texto a analizar:\n\n{texto}"}
        ]

        # Construir la petición a la API de GROQ
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
            st.markdown("### Respuesta del modelo:")
            st.markdown(contenido)