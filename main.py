import uuid
from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from utils import CustomOpenAIEmbeddings
from fastapi import FastAPI, File, UploadFile
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool, StructuredTool, tool
load_dotenv()

@tool
def check_pdf(query: str) -> str:
    """Look up things if required from available PDFs. Answer any generic questions if you can. 
    If you are unsure in any way, use this tool."""
    print(query)
    db = Chroma(persist_directory="./chroma_db", embedding_function=CustomOpenAIEmbeddings())
    docs = db.similarity_search(query)
    return docs[0].page_content

tools = [check_pdf]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

app = FastAPI()

@app.post("/add-document")
async def add_document(
    document: UploadFile = File(...)
):  
    print("FILE TYPE: ", document.content_type)
    out_file_path = "received.pdf"
    with open(out_file_path, "wb") as out_file:
        content = await document.read()  # Read the content of the uploaded file
        out_file.write(content)
    
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader    
    loader = PyPDFLoader(out_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    Chroma.from_documents(documents = chunked_documents, embedding = CustomOpenAIEmbeddings(), persist_directory="./chroma_db")
    return {"message": "Document added successfully", "status": "success"}

@app.post("/chat")
async def chat(
    query: str
):
    response = agent_executor.invoke({"input": query})
    return {"message": "Chat complete", "status": "success", "response": response["output"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)