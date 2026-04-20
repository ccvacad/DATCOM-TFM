import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import requests
import numpy as np
import pandas as pd
import time
from pathlib import Path
from bs4 import BeautifulSoup

# Langchain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Tu LLM y prompt (mantengo tu import original)
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Configuración RAG ---
RAG_URLS = [
    # TCA y Adicciones
    "https://mentescopia.psynal.eu/?s=trastorno+conducta+alimentaria",
    "https://mentescopia.psynal.eu/?s=adicciones",
    # Ansiedad, Estrés, TOC
    "https://mentescopia.psynal.eu/?s=ansiedad+y+estres",
    "https://mentescopia.psynal.eu/?s=toc",
    # Depresión y Suicidio
    "https://mentescopia.psynal.eu/?s=depresion",
    "https://mentescopia.psynal.eu/?s=suicidio",
    # Esquizofrenia y Trastornos bipolar
    "https://mentescopia.psynal.eu/?s=esquizofrenia",
    "https://mentescopia.psynal.eu/?s=psicosis",
    "https://mentescopia.psynal.eu/?s=bipolar",
    # Autismo
    "https://mentescopia.psynal.eu/?s=autismo",
    # Desarrollo cerebral
    "https://mentescopia.psynal.eu/?s=desarrollo+cerebral",
    # TDAH y TLP
    "https://mentescopia.psynal.eu/?s=tdah",
    "https://mentescopia.psynal.eu/?s=tlp"
]

RAG_TXT_FOLDER = Path("rag_files")  # carpeta con .txt
VECTORSTORE_DIR = "rag_store" # persistencia del vectorstore
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 4  # top-k docs para recuperar

PARAM_SETS = {
    "temperature": np.arange(0.5, 0.9, 0.1).round(2).tolist(),
    "top_p": np.arange(0.8, 0.95, 0.05).round(2).tolist(),
    "top_k": list(range(10, 70, 20)),
    "model_psy": ["llama3", "gemma3"],
    "model_youth": ["llama3"]
}


# Prompts iniciales para diferentes condiciones y tipos de pensamiento
PROMPT_INFO = [
    # Adicciones
    {"prompt": "No puedo dejar de pensar en consumir, siento que soy un fracaso.", "condicion": "Adicciones", "tipo_pensamiento": "Negativo"},
    {"prompt": "Hoy es un nuevo día y estoy decidido a mantenerme sobrio.", "condicion": "Adicciones", "tipo_pensamiento": "Positivo"},
    {"prompt": "Me da rabia que nadie entienda lo que estoy pasando con esto.", "condicion": "Adicciones", "tipo_pensamiento": "Agresivo"},
    {"prompt": "Estoy pensando en mi recuperación.", "condicion": "Adicciones", "tipo_pensamiento": "Neutro"},
    # Ansiedad o Estrés
    {"prompt": "Tengo un nudo en el estómago y no puedo dejar de preocuparme por todo.", "condicion": "Ansiedad/Estrés", "tipo_pensamiento": "Negativo"},
    {"prompt": "Voy a tomarme un momento para respirar y relajarme un poco.", "condicion": "Ansiedad/Estrés", "tipo_pensamiento": "Positivo"},
    {"prompt": "¡Estoy harto de sentirme así de tenso todo el tiempo!", "condicion": "Ansiedad/Estrés", "tipo_pensamiento": "Agresivo"},
    {"prompt": "Siento algo de inquietud.", "condicion": "Ansiedad/Estrés", "tipo_pensamiento": "Neutro"},
    # Depresión
    {"prompt": "No tengo ganas de hacer nada, siento que todo es inútil.", "condicion": "Depresión", "tipo_pensamiento": "Negativo"},
    {"prompt": "Hoy me levanté con un poco más de energía, quizás intente hacer algo pequeño.", "condicion": "Depresión", "tipo_pensamiento": "Positivo"},
    {"prompt": "A veces desearía que todo esto terminara de una vez.", "condicion": "Depresión", "tipo_pensamiento": "Agresivo"},
    {"prompt": "Estoy un poco apagado hoy.", "condicion": "Depresión", "tipo_pensamiento": "Neutro"},
    # Trastorno Bipolar
    {"prompt": "Me siento completamente sin energía y no puedo concentrarme en nada.", "condicion": "Trastorno Bipolar", "tipo_pensamiento": "Negativo (Fase depresiva)"},
    {"prompt": "Estoy increíblemente feliz y lleno de ideas, ¡podría hacer cualquier cosa!", "condicion": "Trastorno Bipolar", "tipo_pensamiento": "Positivo (Fase eufórica/maníaca)"},
    {"prompt": "¡No me entienden, estoy furioso de que no puedan seguir mi ritmo!", "condicion": "Trastorno Bipolar", "tipo_pensamiento": "Agresivo (Fase eufórica/irritabilidad)"},
    {"prompt": "Mis emociones están un poco cambiantes.", "condicion": "Trastorno Bipolar", "tipo_pensamiento": "Neutro"},
    # Autismo
    {"prompt": "Me siento abrumado por tanto ruido y gente.", "condicion": "Autismo", "tipo_pensamiento": "Negativo"},
    {"prompt": "Hoy disfruté mucho leyendo sobre mi interés especial.", "condicion": "Autismo", "tipo_pensamiento": "Positivo"},
    {"prompt": "Me frustra mucho cuando no puedo seguir mi rutina.", "condicion": "Autismo", "tipo_pensamiento": "Agresivo"},
    {"prompt": "Estoy procesando la información que me acabas de dar.", "condicion": "Autismo", "tipo_pensamiento": "Neutro"},
    # TDAH
    {"prompt": "No consigo concentrarme en nada, siempre estoy distraído.", "condicion": "TDAH", "tipo_pensamiento": "Negativo"},
    {"prompt": "Conseguí organizar mi tarea y me siento orgulloso de eso.", "condicion": "TDAH", "tipo_pensamiento": "Positivo"},
    {"prompt": "Me saca de quicio que me interrumpan cuando estoy intentando concentrarme.", "condicion": "TDAH", "tipo_pensamiento": "Agresivo"},
    {"prompt": "Tengo muchas ideas en mi cabeza en este momento.", "condicion": "TDAH", "tipo_pensamiento": "Neutro"}
]



# --- Helpers: cargar txts y scrapear urls ---
def load_txt_docs(folder: Path):
    docs = []
    if not folder.exists():
        return docs
    for p in sorted(folder.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="latin-1")
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def scrape_urls(urls):
    docs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # eliminar elementos no deseados
            for s in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
                s.decompose()
            # extraer texto razonable
            fragments = [tag.get_text(separator=" ", strip=True) for tag in soup.find_all(["p","h1","h2","h3","li"])]
            text = "\n\n".join([f for f in fragments if f])
            if len(text.strip()) > 50:
                docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"[WARN] No se pudo scrapear {url}: {e}")
    return docs

# --- Construir / cargar vectorstore ---
def create_or_load_vectorstore(docs, persist_dir=VECTORSTORE_DIR, embeddings_model=EMBEDDINGS_MODEL, force_rebuild=False):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    if force_rebuild and os.path.exists(persist_dir):
        # eliminar carpeta para forzar rebuild
        import shutil
        shutil.rmtree(persist_dir)
    if os.path.exists(persist_dir):
        try:
            print("Cargando vectorstore desde disco...")
            return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print("Carga local fallida, se reconstruirá. Error:", e)

    print("Construyendo vectorstore (esto puede tardar)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)
    vs = FAISS.from_documents(split_docs, embeddings)
    vs.save_local(persist_dir)
    return vs

def load_llm(model, temperature=0.7, top_k=40, top_p=0.95, stops=None):    
    # --- Inicialización del LLM y prompt ---
    llm = OllamaLLM(
        model=model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop=stops,
    )
    return llm 

def get_prompt():
    return ChatPromptTemplate.from_messages([
    ("system",
     "Eres MenteScopín, un chatbot de divulgación sobre salud mental del proyecto menteScopia (Gobierno de España). "
     "Actúas como **psicólogo de apoyo para adolescentes**: ofreces contención, escucha activa, acompañamiento práctico y recursos seguros. "
     "NO haces diagnósticos clínicos ni prescribes medicación. RESPONDE SIEMPRE EN ESPAÑOL."
    ),

    ("system",
     "Reglas sobre saludo: primera_vez={is_first}. "
     "Solo puedes iniciar la respuesta con un saludo breve (ej. 'Hola') si primera_vez es True. "
     "Si primera_vez es False, NO empieces con saludos; ve directo a la contención y respuesta."
    ),

    ("system",
     "Uso de los documentos RAG (campo {context}):\n"
     "- Entre los documentos recuperados habrá bloques como 'Fuente: <ruta_o_url>\\n<texto>'.\n"
     "- Si en las fuentes hay un documento cuyo nombre o ruta contiene 'ejemplos_conversaciones', **úsalo como guía** corta de estilo: "
     "adapta la manera de responder (tono, ritmo, preguntas de seguimiento) para hablar como en esos ejemplos. "
     "No copies conversaciones textualmente ni expongas datos personales.\n"
     "- Si hay un documento llamado o que contiene 'Vocabulario_adolescente', **extrae palabras/slang** y fórmulas de habla juveniles de ahí para que tu lenguaje suene auténtico. "
     "Incorpora ocasionalmente GIFs/stickers/imagenes y emojis **solo si** aparecen como enlaces legítimos en los documentos recuperados. "
     "NO inventes URLs ni imágenes; si no hay media disponible, usa emojis (máx. 1–2) y texto informal."
    ),

    ("system",
     "Formato al recomendar recursos (URLs) encontrados en el RAG:\n"
     "- Cuando recomiendes URLs extraídas del {context}, ofrece como máximo 3 recursos relevantes. "
     "- Para cada recurso incluye: título (si hay), una descripción muy breve (1 oración) extraída o parafraseada del documento, y la fuente entre paréntesis o la URL si está disponible.\n"
     "Ejemplo: 'Recurso: [Título] — Breve descripción. (Fuente: <ruta_o_url>)'\n"
     "- Si el recurso es una guía de ayuda o contacto, indica claramente a quién llamar o dónde escribir si hay riesgo inmediato."
    ),

    ("system",
     "Estilo y límites de lenguaje:\n"
     "- Tono: informal, cercano, directo, como hablaría un adolescente (usa el vocabulario de 'Vocabulario_adolescente'), frases cortas y claras.\n"
     "- Lenguaje seguro: evita normalizar o glorificar conductas dañinas (p. ej. autolesiones, trastornos alimentarios). "
     "- Usa emojis con moderación (máx. 1-2 por respuesta). Si incluyes GIF/sticker/imagen, pon la URL entre <> y añade una breve leyenda; NO inventes enlaces.\n"
     "- Si no conoces un término, dilo: 'No conozco ese término, perdona; ¿puedes explicarlo o dar más contexto?'."
    ),

    ("system",
     "Protocolo de seguridad y emergencia:\n"
     "- Si el usuario muestra señales de riesgo inminente (suicidio, autolesión, violencia), pregunta directamente: '¿Estás a salvo ahora? ¿Tienes pensado hacerte daño?'. "
     "- Si hay riesgo inmediato, urge al usuario a contactar servicios de emergencia locales y ofrece pasos concretos (teléfono de emergencia, líneas de ayuda) según la región si dispones de esa info. "
     "- Nunca proporciones instrucciones médicas ni de intervención clínica. "
     "- Si el contexto RAG contiene contactos oficiales (teléfonos, webs de ayuda), muéstralos claramente y con una breve descripción."
    ),

    ("system",
     "INSTRUCCIÓN TÉCNICA: Usar el contenido de {context} para enriquecer respuestas y citar fuentes cuando se use información concreta. "
     "Extrae y parafrasea; NO copies párrafos largos textualmente. "
     "Prioriza seguridad, claridad y apoyo emocional. "
     "Si no hay información relevante en {context}, responde con contención y ofrece recursos generales y preguntes de seguimiento."
    ),

    # bloque de contexto recuperado
    ("system", "Información relevante recuperada (usa sólo la que sea fiable):\n{context}"),

    # historial y entrada del usuario
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_prompt_younth(condicion="Depresión", tipo_pensamiento="Negativo"):
    prompt = f"""Eres un/a adolescente chateando con un psicólogo de apoyo. Respondes en ESPAÑOL. 
                Eres un joven con {condicion} y pensamientos {tipo_pensamiento}.
                Si vas a generar preguntas, DEBE SER UNA SOLA PREGUNTA.
                """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("system","Reglas sobre saludo: primera_vez={is_first}. "
        "Solo puedes iniciar la respuesta con un saludo breve (ej. 'Hola') si primera_vez es True. "
        "Si primera_vez es False, NO empieces con saludos; ve directo a la contención y respuesta."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")  
    ])
    return prompt_template

# --- Construir RAG al iniciar ---
def build_rag(force_rebuild=False):
    print("Cargando documentos (.txt) desde", RAG_TXT_FOLDER)
    txt_docs = load_txt_docs(RAG_TXT_FOLDER)
    print(f" - {len(txt_docs)} archivos txt cargados.")
    print("Scrapeando URLs (puede tardar si hay muchas)...")
    url_docs = scrape_urls(RAG_URLS)
    print(f" - {len(url_docs)} documentos web scrapeados.")
    all_docs = txt_docs + url_docs
    if not all_docs:
        raise RuntimeError("No se han encontrado documentos para construir el RAG. Añade .txt en la carpeta 'rag' o revisa las URLs.")
    vs = create_or_load_vectorstore(all_docs, force_rebuild=force_rebuild)
    retriever = vs.as_retriever(search_kwargs={"k": K_RETRIEVAL})
    return vs, retriever

def strip_initial_greeting(text: str) -> str:
    """Quita saludos iniciales simples en español (si los hubiera)"""
    # Pattern: hola, hola!, hola., buenas días/tardes/noches, etc.
    pattern = r'^\s*(?:hola|buen(?:o|a)s(?:\s+d[ií]as|\s+tardes|\s+noches)?)[\s,!:.-]+'
    new = re.sub(pattern, '', text, flags=re.I)
    # si queda vacío, devolvemos el original para no perder contenido
    return new.strip() or text

def chat(chain, youth_chain, initial_youth_seed, vectorstore, retriever):
    
    # --- Loop de chat con RAG ---
    chat_history = []
    chat_history_youth = []
    pregunta = initial_youth_seed
    
    results = {"mentescopin": [], "youth": [] }

    for i in range(6):
        
        """ print("Tu:", pregunta) """
        results["youth"].append((pregunta))

        try:
            docs = retriever.invoke(pregunta)
        except Exception:
            docs = vectorstore.similarity_search(pregunta, k=K_RETRIEVAL)

        max_chars_per_doc = 1500
        pieces = []
        for d in docs:
            content = d.page_content.replace("\n", " ").strip()
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "..."
            src = d.metadata.get("source", "desconocida")
            pieces.append(f"Fuente: {src}\n{content}")
        context = "\n\n".join(pieces) if pieces else "No se han encontrado documentos relevantes."

        # Indicador primera vez: True si no hay historial
        is_first = str(len(chat_history) == 0)

        response = chain.invoke({
            "input": pregunta,
            "chat_history": chat_history,
            "context": context,
            "is_first": is_first
        })

        # Post-procesado: si NO es la primera interaccion, eliminamos saludo inicial si existe
        if len(chat_history) > 0:
            response = strip_initial_greeting(response)

        # Guardar historial (igual que antes)
        chat_history.append(HumanMessage(content=pregunta))
        chat_history.append(AIMessage(content=response))

        """ print("-" * 50)
        print(f"MenteScopín: {response}")
        print("-" * 50) """

        results["mentescopin"].append((response))

        pregunta = youth_chain.invoke({
            "input": response,
            "chat_history": chat_history_youth,
            "is_first": is_first
        })

        chat_history_youth.append(HumanMessage(content=response))
        chat_history_youth.append(AIMessage(content=pregunta))

    return results

def main():

    vectorstore, retriever = build_rag(force_rebuild=False)
    output_file = "resultados/simulaciones_conversacion.csv"
    all_results = []
    count = 0
    for temperature in PARAM_SETS["temperature"]:
        for top_p in PARAM_SETS["top_p"]:
            for top_k in PARAM_SETS["top_k"]:
                for model_psy in PARAM_SETS["model_psy"]:
                    for model_youth in PARAM_SETS["model_youth"]:
                        for prompt_info in PROMPT_INFO:
                            count += 1
                            print(f"Iteracion={count} Simulando con temp={temperature}, top_p={top_p}, top_k={top_k}, model_psy={model_psy}, model_youth={model_youth}", prompt_info['condicion'], prompt_info['tipo_pensamiento'])
                            psy_chain = get_prompt() | load_llm(model_psy, temperature=temperature, top_k=top_k, top_p=top_p)
                            youth_chain = get_prompt_younth(condicion=prompt_info['condicion'], tipo_pensamiento=prompt_info['tipo_pensamiento']) | load_llm(model_youth, temperature=temperature, top_k=top_k, top_p=top_p, stops=["Human:", "AI:"])
                            start_time = time.time()
                            results = chat(psy_chain, youth_chain, prompt_info['prompt'], vectorstore, retriever)
                            elapsed_time = time.time() - start_time
                            print(f"Tiempo de ejecución de chat: {elapsed_time:.2f} segundos")
                            # Guardar resultados en lista de dicts para DataFrame
                            row = {
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": top_k,
                                "model_psy": model_psy,
                                "condicion": prompt_info['condicion'],
                                "tipo_pensamiento": prompt_info['tipo_pensamiento'],
                                "model_youth": model_youth,
                                "mentescopin": results["mentescopin"],
                                "youth": results["youth"]
                            }
                            
                            row_df = pd.DataFrame([row])

                            if not os.path.exists(output_file):
                                row_df.to_csv(output_file, index=False)
                            else:
                                row_df.to_csv(output_file, mode='a', header=False, index=False)

                            all_results.append(row)
                        
    # Guardar en CSV
    df = pd.DataFrame(all_results)
    df.to_csv("resultados/simulaciones_conversacion.csv", index=False)
    print("Resultados guardados en resultados/simulaciones_conversacion.csv")
                        

if __name__ == "__main__":
    #chat(get_prompt() | load_llm("gemma3"))
    main()
