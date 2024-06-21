import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_loader = PyPDFLoader(pdf)
        for page in pdf_loader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding--001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Beantworte die Frage so detailliert wie möglich aus dem Kontext.
    Wenn die Frage sich nicht aus dem Kontext beantworten lässt, antworte "Die Frage kann nicht aus dem Kontext beantwortet werden".
    Kontext:\n{context}\n
    Frage:\n{question}\n

    Antwort:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss-index", embeddings=embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    print(response)
    st.write("Antwort: ", response["output_text"])


def main():
    st.set_page_config("Chat mit Dokumenten")
    st.header("Gemini-powered Chat mit Dokumenten")

    user_question = st.text_input("Stellen Sie eine Frage.")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menü:")
        pdf_docs = st.file_uploader(
            "Wählen Sie die PDF Dokumente aus und klicken Sie auf 'laden'"
        )
        if st.button("laden"):
            with st.spinner("verarbeite ..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
