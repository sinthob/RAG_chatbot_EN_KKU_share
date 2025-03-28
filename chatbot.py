from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4  
from langchain_community.vectorstores import FAISS  # ใช้ FAISS เป็น vector store ชั่วคราว
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os


# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,          # ดึงผลลัพธ์ที่เกี่ยวข้อง 20 อันดับ
        "fetch_k": 50     # เอา 50 อันดับแรกก่อนคัดมา 20
    }
)


# call this function for every message added to the chatbot
def stream_response(message, history, pdf_file):
    combined_knowledge = ""

    # ✅ ถ้ามี PDF แนบมา ให้ฝังเฉพาะใน memory ไม่เพิ่มเข้า vector_store หลัก
    if pdf_file is not None:
        try:
            ext = os.path.splitext(pdf_file.name)[-1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(pdf_file.name)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(pdf_file.name)
            else:
                return history + [[message, f"❌ รองรับเฉพาะไฟล์ PDF และ DOCX (.pdf, .docx) เท่านั้น"]]
            
            raw_documents = loader.load()
            print("[DEBUG] DOCX Content:")
            for doc in raw_documents:
                print(doc.page_content)


            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(raw_documents)
            
            print("[DEBUG] Chunks after splitting:")
            for i, chunk in enumerate(chunks):
                print(f"[Chunk {i}]\n{chunk.page_content}\n{'-'*50}")

            # ✅ ใช้ vectorstore ชั่วคราวใน memory
            temp_vectorstore = FAISS.from_documents(chunks, embeddings_model)
            temp_retriever = temp_vectorstore.as_retriever(search_kwargs={'k': 10})
            docs = temp_retriever.invoke(message)
            
            # ✅ DEBUG: Show which chunks were retrieved for this prompt
            print("[DEBUG] Retrieved Chunks for Prompt:")
            for i, doc in enumerate(docs):
                print(f"[Doc {i}]\n{doc.page_content}\n{'-'*50}")

            print("[DEBUG] ✅ ใช้ข้อมูลจาก PDF ที่อัปโหลด")

        except Exception as e:
            return history + [[message, f"❌ PDF Error: {str(e)}"]]
    else:
        # ✅ ถ้าไม่มี PDF แนบมา ใช้ข้อมูลเดิมจาก database (Chroma)
        docs = retriever.invoke(message)
        print("[DEBUG] 🧠 ใช้ข้อมูลจาก database vector store เดิม")

    combined_knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
    You are an assistant which answers questions based on knowledge which is provided to you.
    While answering, you don't use your internal knowledge, 
    but solely the information in the "The knowledge" section.
    You don't mention anything to the user about the provided knowledge.

    The question: {message}

    Conversation history: {history}

    The knowledge: {combined_knowledge}
    """

    partial_response = ""
    for response in llm.stream(rag_prompt):
        partial_response += response.content

    return history + [[message, partial_response]]



# initiate the Gradio app
with gr.Blocks() as demo:
    with gr.Row():
        input_textbox = gr.Textbox(placeholder="พิมพ์คำถามที่นี่...", label="Prompt", lines=2)
        pdf_input = gr.File(label="แนบไฟล์ (PDF หรือ DOCX)", file_types=[".pdf", ".docx"])
    chatbot = gr.Chatbot(label="Chatbot", height=400, type="tuples")
    send_button = gr.Button("ส่งคำถาม")

    def wrapper(message, pdf_file, history):
        updated_history = stream_response(message, history, pdf_file)
        return updated_history, gr.update(value=""), gr.update(value=None)

    send_button.click(
        fn=wrapper,
        inputs=[input_textbox, pdf_input, chatbot],
        outputs=[chatbot, input_textbox, pdf_input]
    )




# launch the Gradio app
#chatbot.launch()
demo.launch()