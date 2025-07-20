from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import asyncio
from pathlib import Path

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# MistralAI imports
import requests
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Learning Plan RAG Chatbot API", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

class DocumentUploadResponse(BaseModel):
    message: str
    documents_processed: int

# Custom MistralAI Embeddings wrapper
class MistralAIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except Exception as e:
            raise Exception(f"MistralAI Embeddings API error: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

# Custom MistralAI LLM wrapper
class MistralAILLM:
    def __init__(self, api_key: str, model: str = "mistral-large-latest", temperature: float = 0.7, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"MistralAI API error: {str(e)}")

# Global variables
vectorstore = None
qa_chain = None
embeddings = None
llm = None

# Initialize MistralAI models
def initialize_models():
    global embeddings, llm
    
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    if not mistral_api_key:
        print("❌ MISTRAL_API_KEY not found in environment variables")
        embeddings = None
        llm = None
        return
    
    try:
        # Initialize MistralAI LLM
        llm = MistralAILLM(
            api_key=mistral_api_key,
            model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            temperature=float(os.getenv("MISTRAL_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MISTRAL_MAX_TOKENS", "1000"))
        )
        print("✅ MistralAI LLM initialized successfully")
        
        # Initialize MistralAI Embeddings
        embeddings = MistralAIEmbeddings(
            api_key=mistral_api_key,
            model=os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
        )
        print("✅ MistralAI Embeddings initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing MistralAI: {e}")
        embeddings = None
        llm = None

# Initialize models
initialize_models()

def create_custom_prompt():
    """Create a custom prompt template for educational assistant in Russian"""
    template = """Ты умный образовательный помощник, специализирующийся на учебных планах и образовательных программах.
    Используй следующий контекст из образовательных документов для ответа на вопрос студента.
    
    Контекст: {context}
    
    Вопрос: {question}
    
    Инструкции:
    - Отвечай кратко и точно, не более 5 предложений
    - Если вопрос касается конкретных курсов, укажи количество зачетных единиц и семестр, если эта информация доступна
    - Если ты не знаешь ответ на основе предоставленного контекста, четко скажи об этом
    - Форматируй ответ как обычный текст, БЕЗ markdown разметки
    - Не используй звездочки, решетки, дефисы для списков или другие символы разметки
    - Включай соответствующие коды курсов или названия модулей, когда это применимо
    
    Ответ:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# Custom RAG chain for MistralAI
class CustomRAGChain:
    def __init__(self, llm, retriever, prompt_template):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template
    
    def __call__(self, query_dict):
        query = query_dict["query"]
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format the prompt
        formatted_prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        # Generate response using MistralAI
        response = self.llm.generate(formatted_prompt)
        
        return {
            "result": response,
            "source_documents": docs
        }

def process_pdf_documents(pdf_paths: List[str]) -> List[Document]:
    """Process PDF documents and return chunks"""
    documents = []
    
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            documents.extend(pages)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def initialize_vectorstore(documents: List[Document]):
    """Initialize FAISS vectorstore with documents"""
    global vectorstore, qa_chain
    
    if not embeddings:
        raise HTTPException(status_code=500, detail="MistralAI embeddings not configured. Please set MISTRAL_API_KEY environment variable.")
    
    if not llm:
        raise HTTPException(status_code=500, detail="MistralAI LLM not configured. Please set MISTRAL_API_KEY environment variable.")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create custom QA chain with MistralAI
    custom_prompt = create_custom_prompt()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = CustomRAGChain(llm, retriever, custom_prompt)

@app.on_event("startup")
async def startup_event():
    """Initialize the system with any existing PDFs in the documents folder"""
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    pdf_files = list(documents_dir.glob("*.pdf"))
    
    if pdf_files and embeddings:
        try:
            documents = process_pdf_documents([str(f) for f in pdf_files])
            if documents:
                initialize_vectorstore(documents)
                print(f"Initialized with {len(documents)} document chunks from {len(pdf_files)} PDFs")
        except Exception as e:
            print(f"Error during startup initialization: {e}")

@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents"""
    if not embeddings:
        raise HTTPException(status_code=500, detail="MistralAI embeddings not configured. Please set MISTRAL_API_KEY environment variable.")
    
    if not llm:
        raise HTTPException(status_code=500, detail="MistralAI LLM not configured. Please set MISTRAL_API_KEY environment variable.")
    
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    pdf_paths = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        # Save uploaded file
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        pdf_paths.append(str(file_path))
    
    try:
        # Process documents
        documents = process_pdf_documents(pdf_paths)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents could be processed")
        
        # Initialize or update vectorstore
        initialize_vectorstore(documents)
        
        return DocumentUploadResponse(
            message=f"Successfully processed {len(files)} PDF files using MistralAI",
            documents_processed=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages and return AI responses"""
    if not qa_chain:
        raise HTTPException(
            status_code=400, 
            detail="No documents have been uploaded yet. Please upload PDF documents first."
        )
    
    try:
        # Get response from QA chain
        result = qa_chain({"query": message.message})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source_info = doc.metadata.get("source", "Unknown")
                page_info = doc.metadata.get("page", "")
                if page_info:
                    source_info += f" (page {page_info + 1})"
                sources.append(source_info)
        
        return ChatResponse(
            response=result["result"],
            sources=list(set(sources))  # Remove duplicates
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    mistral_configured = bool(mistral_api_key and llm and embeddings)
    
    return {
        "status": "healthy",
        "vectorstore_initialized": vectorstore is not None,
        "mistral_configured": mistral_configured,
        "models": {
            "llm": f"MistralAI ({os.getenv('MISTRAL_MODEL', 'mistral-large-latest')})" if llm else "Not configured",
            "embeddings": f"MistralAI ({os.getenv('MISTRAL_EMBED_MODEL', 'mistral-embed')})" if embeddings else "Not configured"
        },
        "api_key_status": {
            "mistral_api_key": "configured" if mistral_api_key else "missing"
        }
    }

@app.get("/documents")
async def list_documents():
    """List uploaded documents"""
    documents_dir = Path("documents")
    if not documents_dir.exists():
        return {"documents": []}
    
    pdf_files = list(documents_dir.glob("*.pdf"))
    return {
        "documents": [f.name for f in pdf_files],
        "count": len(pdf_files)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)