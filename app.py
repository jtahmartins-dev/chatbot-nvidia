import os
import streamlit as st
from dotenv import load_dotenv

#carrega as variaveis de ambiente .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

#Configuração da pagina do Streamlit
st.set_page_config(page_title="NVIDIA RAG PDF Asistente",page_icon="🧠",layout="centered")
st.title("Asistende Virtual de Manuais de Produtos e Sistemas em PDF")
st.write("Qual sua dúvida, como posso ajudar ?")

nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

if nvidia_api_key:
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            nvidia_api_key=st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass

if nvidia_api_key:
    st.info("Por favor adicione sua Chave ao arquivo .env ou ao Secrets")
    st.stop()

#carragenado o RAg a partir do pdf - criando a vetorizado do pdf
@st.cache_resource(show_spinner="Processando o PDF...")
def inicializar_rag():
    nome_arquivo="manual.pdf"

    if not os.path.exists("nome_arquivo"):
        st.error(f"Arquivo'{nome_arquivo}' não foi encontrado")
        st.stop
   
    loader = PyPDFLoader(nome_arquivo)
    paginas = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_olverlap=50
    )

    docs=text_splitter.split_documents(paginas)

    embeddings= NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage"
    )

    vectorstore=FAISS.from_documents(docs,embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"K":4})

retriver = inicializar_rag()

llm=ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    nvidia_api_key=nvidia_api_key,
    temperature=0.2

)

template_prompt="""
    Voce é um assistente técnico especializado e prestativo.
    Os textos fragmentados de contexto inseridos abaixo foram extraidos de um manual de produtos e serviços e podem estar  EM INGLÊS.
    Sua tarefa é analizar o contexto ainda que esteja em ingles, mas sempre responder a pergunta do usuario OBRIGATORIAMENTE EM PORTUGUÊS DO BRASIL.
   
    Use estritamente as informações fornecidas para responder. Se a resposta não puder ser encontrada no texto, diga Explicitamente: "Desculpe, mas essa informação não consta no manual "

    Contexto (em ingles):
    {context}

    Pergunta (em português): question
    Resposta em prtuguês:

"""
prompt = ChatPromptTemplate.from_template(template_prompt)

#pipeline do RAG para evitar alucinações " pro chat não viajar "
rag_chain=(
    {"context":retriver,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()

)

if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"assistant","content":"Olá! Processei manual em inglês com sucesso. O que você deseja saber ?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt_usuario:=st.chat_input("Ex: Qual o significado do codigo 4"):
    st.session_state.messages.append({"role":"user","content":prompt_usuario})
    with st.chat_message("user"):
        st.write(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando o manual técnico...."):
            try:
                resposta=rag_chain.invoke(prompt_usuario)
                st.write(resposta)
                st.session_state.messages({"role":"assistant","content":resposta})
            except Exception as e:
                st.error(f"Erro ao processar a requisição da APi: {e} ")