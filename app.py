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
            st.info("Cargando datos...")
            return FAISS.load_local(
                index_path,
                HuggingFaceEmbeddings(model_name=embeddings_model),
                allow_dangerous_deserialization=True
            )
        
        st.info("Generando datos...")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("Error en el procesamiento de la información.")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
        split_docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        vectorstore.save_local(index_path)
        st.success("Datos procesados correctamente!")
        return vectorstore
    
    except Exception as e:
        st.error("Error al procesar la información.")
        raise

def main():
    st.set_page_config(page_title="EcoAsistence", page_icon="🌿", layout="wide")
    
    with st.sidebar:
        st.markdown("""
            <h2 style='text-align: left; font-family: Arial, sans-serif; color: #B0B0B0;'>EcoAsistence</h2>
        """, unsafe_allow_html=True)
        st.header("⚙️ Configuración")
        bot_name = st.text_input("Nombre del Bot:", value="EcoAsistente")
        temperature = st.slider("Nivel de Creatividad:", 0.0, 1.0, 0.7)
        llm.temperature = temperature
    
    st.title("🧠 Chatbot Inteligente con RAG")
    
    if not os.path.exists(DEFAULT_PDF):
        st.error("Error en la carga de datos.")
        return
    
    try:
        if "retriever" not in st.session_state:
            vectorstore = get_or_create_faiss_index(DEFAULT_PDF, INDEX_PATH, EMBEDDINGS_MODEL)
            st.session_state.retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    except Exception as e:
        st.error("Error en la configuración del asistente.")
        return
    
    system_prompt = f"""
    Eres {bot_name}, un asistente especializado. Sigue estas reglas:
    1. Usa información del contexto cuando esté disponible
    2. Si no hay contexto relevante, responde con tu conocimiento general
    3. Sé conciso (máximo 3 oraciones)
    4. Haz preguntas de seguimiento cuando sea relevante
    5. Mantén un tono profesional pero amigable
    """
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{system_prompt}\nContexto: {{context}}\nPregunta: {{question}}\nRespuesta:"
    )
    
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
    
    chain = ConversationalRetrievalChain(
        retriever=st.session_state.retriever,
        combine_docs_chain=qa_chain,
        question_generator=question_generator,
        return_source_documents=False,
        verbose=False
    )
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
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
                
                answer = response.get("answer", "Lo siento, no encontré una respuesta adecuada.")
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=answer))
                
                with st.chat_message("ai"):
                    st.write(answer)
        
        except Exception as e:
            st.error("⚠️ Error al procesar tu consulta.")
    
    for msg in st.session_state.chat_history:
        with st.chat_message("human" if isinstance(msg, HumanMessage) else "ai"):
            st.write(msg.content)

if __name__ == "__main__":
    main()

