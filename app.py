import config as cfg
import vais
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from google.cloud import aiplatform
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


CHAT_MODEL = ChatGoogleGenerativeAI(model=cfg.TEXT_MODEL, temperature=0.0)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

aiplatform.init(project=cfg.PROJECT_ID, location=cfg.REGION)


def embed_files(file_paths):
    paths = [file.name for file in file_paths]
    for path in paths:
        text = embed_docs_from_pdf_path(path)


def embed_docs_from_pdf_path(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunk_docs = TEXT_SPLITTER.split_documents(docs)
    texts = [doc.page_content for doc in chunk_docs]
    metadatas = [doc.metadata for doc in chunk_docs]
    vais.embed(texts, metadatas)



def get_conversational_chain():
    prompt = """
    Beantworte die Frage so detailliert wie möglich aus dem Kontext.
    Wenn die Frage sich nicht aus dem Kontext beantworten lässt, antworte mit "Die Frage kann nicht aus dem Kontext beantwortet werden".
    Kontext:\n{context}\n
    Frage:\n{question}\n
    Antwort:
    """

    prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

    chain = load_qa_chain(CHAT_MODEL, chain_type="stuff", prompt=prompt)
    return chain


def search(question):
    result = vais.search(question)
    return result

def main():
    with gr.Blocks() as demo:
        file_output = gr.File()
        upload_button = gr.UploadButton(
            "click to upload a file",
            file_count="multiple",
        )
        upload_button.upload(fn=embed_files, inputs=upload_button, outputs=file_output)
        question = gr.TextArea(label="question", lines=8)
        submit = gr.Button("submit")
        answer = gr.TextArea(label="answer", lines=8)
        status = gr.Text(label="status", value="")
        submit.click(fn=search, inputs=question, outputs=answer)
    demo.launch()


if __name__ == "__main__":
    main()
