from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
import logging
from typing import List
from langchain.schema import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL.upper())

openai_embeddings_model = None

def get_openai_embeddings():
    global openai_embeddings_model
    if not openai_embeddings_model:
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not configured. Cannot initialize OpenAIEmbeddings.")
            raise ValueError("OPENAI_API_KEY not set")
        try:
            openai_embeddings_model = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAIEmbeddings model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbeddings: {e}")
            raise
    return openai_embeddings_model

def load_document(file_path: str, content_type: str) -> List[Document]:
    logger.info(f"Attempting to load document from {file_path} with content_type: {content_type}")
    try:
        if content_type == "text/plain":
            loader = TextLoader(file_path, encoding="utf-8")
        elif content_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": # DOCX
            loader = Docx2txtLoader(file_path)
        else:
            logger.error(f"Unsupported content_type for document loading: {content_type} for file {file_path}")
            raise ValueError(f"Unsupported content type: {content_type}")

        documents = loader.load()
        logger.info(f"Successfully loaded document from {file_path} using {loader.__class__.__name__}. Pages/Docs: {len(documents)}")
        return documents
    except ValueError as ve:
        logger.error(f"Error loading document {file_path}: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error loading document from {file_path} with loader for {content_type}: {e}")
        return []

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks

def get_document_embeddings(chunks: List[Document]) -> List[List[float]]:
    embeddings_model = get_openai_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    try:
        embeddings = embeddings_model.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks.")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating document embeddings: {e}")
        # Potentially re-raise or handle to prevent partial processing
        raise
