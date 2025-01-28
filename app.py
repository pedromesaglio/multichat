import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
import os


llm = Ollama(
    model="deepseek-r1:8b",
    temperature=0.7,
    system="Responde como un experto amigable con logica basica y cuando lo necesites usando el contexto proporcionado."
)


INDEX_PATH = os.path.abspath("./faiss_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PDF = "documents/medioambiente.pdf"

def get_or_create_faiss_index(pdf_path, index_path, embeddings_model):
    try:
        if os.path.exists(index_path):
            st.info("Cargando índice FAISS desde disco...")
            return FAISS.load_local(
                index_path,
                HuggingFaceEmbeddings(model_name=embeddings_model),
                allow_dangerous_deserialization=True
            )
        
        st.info("Generando nuevo índice FAISS...")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Archivo PDF no encontrado: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        split_docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        vectorstore.save_local(index_path)
        st.success("Índice FAISS generado correctamente!")
        return vectorstore
    
    except Exception as e:
        st.error(f"Error procesando PDF: {str(e)}")
        raise

def main():
    st.title("🧠 Chatbot Inteligente con RAG")
    
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        bot_name = st.text_input("Nombre del Bot:", value="EcoAsistente")
        temperature = st.slider("Nivel de Creatividad:", 0.0, 1.0, 0.7)
        llm.temperature = temperature
    
    
    if not os.path.exists(DEFAULT_PDF):
        st.error(f"🔍 Por favor coloca tu PDF en: {os.path.abspath(DEFAULT_PDF)}")
        return
    
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    
    try:
        if "retriever" not in st.session_state:
            vectorstore = get_or_create_faiss_index(DEFAULT_PDF, INDEX_PATH, EMBEDDINGS_MODEL)
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4}
            )
    except Exception as e:
        st.error(f"🚨 Error inicial: {str(e)}")
        return
    

    system_prompt = f"""
    Eres {bot_name}, un asistente especializado. Sigue estas reglas:
    1. Usa exclusivamente información del contexto proporcionado
    2. Sé conciso (máximo 3 oraciones)
    3. Haz preguntas de seguimiento cuando sea relevante
    4. Mantén un tono profesional pero amigable
    """
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{system_prompt}\nContexto: {{context}}\nPregunta: {{question}}\nRespuesta:"
    )
    
    # Configuración de las cadenas
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
    
    # --- CORRECCIÓN DEL ERROR ---
    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=False
    )
    
    chain = ConversationalRetrievalChain(
        retriever=st.session_state.retriever,
        combine_docs_chain=qa_chain,
        question_generator=question_generator,  # Componente requerido
        return_source_documents=True,
        verbose=False
    )
    # ----------------------------
    
    # Interfaz de chat
    user_input = st.chat_input("Escribe tu mensaje aquí...")
    
    if user_input:
        if user_input.lower() in ["salir", "adios", "exit"]:
            st.success("¡Gracias por charlar! Hasta pronto. 👋")
            st.stop()
        
        try:
            with st.spinner("🔍 Analizando y generando respuesta..."):
                response = chain({
                    "question": user_input,
                    "chat_history": [msg.content for msg in st.session_state.chat_history]
                })
            
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            
            with st.expander("📄 Documentos de referencia utilizados"):
                for doc in response["source_documents"]:
                    st.markdown(f"""
                    **Página {doc.metadata['page'] + 1}**  
                    {doc.page_content[:250]}...
                    """)
                    
        except Exception as e:
            st.error(f"⚠️ Error al procesar tu consulta: {str(e)}")
    
    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("human"):
                st.write(msg.content)
        else:
            with st.chat_message("ai"):
                st.write(msg.content)

if __name__ == "__main__":
    main()