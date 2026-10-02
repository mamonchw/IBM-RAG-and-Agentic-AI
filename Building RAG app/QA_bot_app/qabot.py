from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')



def get_llm():
    llm = Ollama(
        model="gemma2:2b", # Replace with your local model name (e.g., "llama2", "mistral")
        temperature=0.5,
        num_predict=256
    )
    return llm
    
def document_loader(file):
    loader=PyPDFLoader(file)
    loaded_document=loader.load()
    return loaded_document

def text_splitter(documents):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks=splitter.split_documents(documents)
    return chunks

def vector_database(chunks):
    embedding_model=OllamaEmbeddings(model="embeddinggemma:latest")
    vector_db= Chroma.from_documents(chunks,embedding_model)
    return vector_db

def retriever(file):
    splits=document_loader(file)
    chunks=text_splitter(splits)
    vector_db=vector_database(chunks)
    retriever=vector_db.as_retriever()
    return retriever


#QA Chain
def retriever_qa(file, query):
    llm=get_llm()
    retriever_obj=retriever(file)
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )
    response=qa.invoke(query)
    return response["result"]


rag_application=gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query" ,lines=2, placeholder="Type your question here...")       
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)


rag_application.launch(server_name="0.0.0.0",server_port=7860, share=True)


    