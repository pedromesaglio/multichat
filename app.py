import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import time
import hashlib

# Configuración inicial
INDEX_PATH = os.path.abspath("./faiss_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PDF = "documents/medioambiente.pdf"
CHAT_HISTORY_LIMIT = 20

def get_pdf_hash(pdf_path):
    """Calcula el hash del PDF para detectar cambios"""
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def get_or_create_faiss_index(pdf_path, index_path, embeddings_model):
    try:
        current_hash = get_pdf_hash(pdf_path) if os.path.exists(pdf_path) else ""
        hash_file = os.path.join(index_path, "pdf_hash.md5")
        
        if os.path.exists(index_path) and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                stored_hash = f.read()
            
            if stored_hash == current_hash:
                st.info("Cargando datos desde el índice existente...")
                return FAISS.load_local(
                    index_path,
                    HuggingFaceEmbeddings(model_name=embeddings_model),
                    allow_dangerous_deserialization=True
                )
        
        st.info("Procesando documentos y generando nuevo índice...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        vectorstore.save_local(index_path)
        with open(os.path.join(index_path, "pdf_hash.md5"), "w") as f:
            f.write(current_hash)
        
        st.success("Índice generado exitosamente!")
        return vectorstore
    
    except Exception as e:
        st.error(f"Error en el procesamiento de documentos: {str(e)}")
        raise

def init_session_state():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None
    
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "temperature": 0.7,
            "system_prompt": """Eres EcoAsistence, un asistente especializado. Sigue estas reglas:
1. Usa información del contexto cuando esté disponible
2. Si no hay contexto relevante, responde con conocimiento general
3. Sé conciso (máximo 3 oraciones)
4. Haz preguntas de seguimiento cuando sea relevante
5. Mantén un tono profesional pero amigable"""
        }

def setup_sidebar():
    with st.sidebar:
        st.markdown("""
            <h2 style='text-align: center; color: #4CAF50; margin-bottom: 30px;'>
            🌿 EcoAsistence
            </h2>
        """, unsafe_allow_html=True)
        
        # Gestión de chats
        st.subheader("💬 Chats")
        if st.button("➕ Nuevo Chat", use_container_width=True):
            new_chat_id = f"Chat_{int(time.time())}"
            st.session_state.current_chat = new_chat_id
            st.session_state.chat_sessions[new_chat_id] = []
        
        if st.session_state.chat_sessions:
            selected_chat = st.selectbox(
                "Chats disponibles:",
                options=list(st.session_state.chat_sessions.keys()),
                index=0,
                key="chat_selector"
            )
            st.session_state.current_chat = selected_chat
        
        # Configuración
        st.subheader("⚙️ Configuración")
        st.session_state.llm_config["temperature"] = st.slider(
            "Nivel de Creatividad:",
            0.0, 1.0, 0.7,
            help="Controla la aleatoriedad de las respuestas (0 = preciso, 1 = creativo)"
        )

def render_chat_messages():
    if st.session_state.current_chat:
        messages = st.session_state.chat_sessions[st.session_state.current_chat]
        for msg in messages[-CHAT_HISTORY_LIMIT:]:
            with st.chat_message("user" if isinstance(msg, HumanMessage) else "ai"):
                st.markdown(f"""
                    <div style='padding: 12px;
                        border-radius: 8px;
                        background-color: #000000;
                        color: #FFFFFF;
                        margin: 8px 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        border-left: { "3px solid #4CAF50" if isinstance(msg, AIMessage) else "none" }'>
                        {msg.content}
                    </div>
                """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="EcoAsistence",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado para el input
    st.markdown("""
        <style>
        .stChatInput textarea {
            color: #FFFFFF !important;
            background-color: #000000 !important;
        }
        .stChatInput button {
            color: #000000 !important;
            background-color: #4CAF50 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    setup_sidebar()
    
    st.title("🌍 Asistente de Medio Ambiente Inteligente")
    st.caption("Pregunta cualquier cosa sobre documentos medioambientales")
    
    # Verificar existencia del PDF
    if not os.path.exists(DEFAULT_PDF):
        st.error("Error: Archivo PDF no encontrado en la ruta especificada")
        return
    
    # Inicializar vectorstore
    try:
        if "retriever" not in st.session_state:
            with st.spinner("Inicializando sistema..."):
                vectorstore = get_or_create_faiss_index(
                    DEFAULT_PDF, INDEX_PATH, EMBEDDINGS_MODEL
                )
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10}
                )
    except Exception as e:
        st.error(f"Error de inicialización: {str(e)}")
        return
    
    # Inicializar cadena conversacional
    if "chain" not in st.session_state:
        llm = Ollama(
            model="deepseek-r1:8b",
            temperature=st.session_state.llm_config["temperature"],
            system=st.session_state.llm_config["system_prompt"]
        )
        
        qa_prompt = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template="""
            {system_prompt}
            
            Contexto relevante:
            {context}
            
            Pregunta: {question}
            Respuesta útil:
            """.strip()
        )
        
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.retriever,
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "prompt": qa_prompt,
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}"
                )
            },
            return_source_documents=True,
            verbose=False
        )

    # Manejar entrada de usuario
    user_input = st.chat_input("Escribe tu pregunta sobre medio ambiente...")
    if user_input:
        if user_input.lower() in ["salir", "adios", "exit"]:
            st.success("¡Gracias por usar EcoAsistence! Hasta pronto. 🌱")
            st.stop()
        
        try:
            # Agregar mensaje de usuario al historial
            human_msg = HumanMessage(content=user_input)
            if st.session_state.current_chat is None:
                st.session_state.current_chat = f"Chat_{int(time.time())}"
                st.session_state.chat_sessions[st.session_state.current_chat] = []
            
            st.session_state.chat_sessions[st.session_state.current_chat].append(human_msg)
            
            with st.spinner("🔍 Analizando y generando respuesta..."):
                response = st.session_state.chain({
                    "question": user_input,
                    "system_prompt": st.session_state.llm_config["system_prompt"],
                    "chat_history": [
                        (msg.content, AIMessage(content=msg.content).content) 
                        for msg in st.session_state.chat_sessions[st.session_state.current_chat] 
                        if isinstance(msg, AIMessage)
                    ]
                })
                
                answer = response.get("answer", "No pude encontrar una respuesta adecuada.")
                ai_msg = AIMessage(content=answer)
                st.session_state.chat_sessions[st.session_state.current_chat].append(ai_msg)
        
        except Exception as e:
            st.error(f"Error al procesar la consulta: {str(e)}")
            st.session_state.chat_sessions[st.session_state.current_chat].pop()
    
    render_chat_messages()

if __name__ == "__main__":
    main()