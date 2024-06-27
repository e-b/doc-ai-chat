import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_vertexai import VertexAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


PROJECT="genai-lab-390210"
LOCATION="europe-west3"

#PERSIST_PATH = "./chromadb/"
TEXT_MODEL = "gemini-1.5-pro"
EMBEDDING_MODEL = "textembedding-gecko@003"

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


model = ChatGoogleGenerativeAI(model=TEXT_MODEL, temperature=0.0)

EMBEDDING_NUM_BATCH = 5

embeddings = VertexAIEmbeddings(
    model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_NUM_BATCH
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def embed_files(file_paths):
    paths = [file.name for file in file_paths]
    for path in paths:
        text = text_from_pdf(path)
        chunks = chunks_from_text(text)
        faiss_embed(chunks)


def text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = ""
    for doc in docs:
        print(doc.page_content)
        text += doc.page_content
    return text


def chunks_from_text(text):
    chunks = text_splitter.split_text(text)
    return chunks


def faiss_embed(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt = """
    Beantworte die Frage so detailliert wie möglich aus dem Kontext.
    Wenn die Frage sich nicht aus dem Kontext beantworten lässt, antworte mit "Die Frage kann nicht aus dem Kontext beantwortet werden".
    Kontext:\n{context}\n
    Frage:\n{question}\n

    Antwort:
    """

    prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def ask(user_question):
    db = FAISS.load_local("faiss-index", embeddings=embeddings)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    print(response)


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
        submit.click(fn=ask, inputs=question, outputs=answer)
    demo.launch()


if __name__ == "__main__":
    main()
