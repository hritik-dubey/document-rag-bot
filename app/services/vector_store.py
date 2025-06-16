from qdrant_client import QdrantClient, models
from app.core.config import settings
import logging
from typing import List, Dict, Any
from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL.upper())

qdrant_client = None
vector_size = 1536 # Default for OpenAI Ada v2, make configurable later if needed

def get_qdrant_client() -> QdrantClient:
    global qdrant_client
    if qdrant_client is None:
        try:
            qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=20 # seconds for timeout
            )
            # Test connection / readiness
            qdrant_client.health_check()
            logger.info(f"Successfully connected to Qdrant at {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {settings.QDRANT_URL}: {e}")
            qdrant_client = None # Ensure it's None if connection failed
            # Depending on strictness, could raise an exception here
    return qdrant_client

def create_collection_if_not_exists(collection_name: str, vector_size: int = vector_size, distance: models.Distance = models.Distance.COSINE):
    client = get_qdrant_client()
    if not client:
        logger.error("Qdrant client not available. Cannot create collection.")
        return
    try:
        # Check if collection exists
        try:
            client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists in Qdrant.")
            return # Collection exists, no need to create
        except Exception as e:
            # Assuming error means collection does not exist (this could be more specific)
            # Qdrantpy_client.http.exceptions.UnexpectedResponse: Unexpected Response: 404 Not Found  Detail: Not found: Collection `collection_name` doesn't exist!
            logger.info(f"Collection '{collection_name}' does not exist. Attempting to create it.")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )
        logger.info(f"Successfully created collection '{collection_name}' in Qdrant with vector size {vector_size} and distance {distance}.")
    except Exception as e:
        logger.error(f"Failed to create or verify collection '{collection_name}' in Qdrant: {e}")

# Example of how to ensure collection exists on startup (optional, can be called from main.py or a startup event)
# def ensure_default_collection():
# create_collection_if_not_exists(settings.QDRANT_COLLECTION_NAME, vector_size)


def add_documents_to_qdrant(collection_name: str, chunks: List[Document], embeddings: List[List[float]], metadata_list: List[Dict[str, Any]] | None = None):
    client = get_qdrant_client()
    if not client:
        logger.error("Qdrant client not available. Cannot add documents.")
        return None

    if not chunks or not embeddings:
        logger.warning("No chunks or embeddings provided to add to Qdrant.")
        return None

    if len(chunks) != len(embeddings):
        logger.error(f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}).")
        raise ValueError("Chunks and embeddings count mismatch")

    if metadata_list and len(chunks) != len(metadata_list):
        logger.error(f"Mismatch between number of chunks ({len(chunks)}) and metadata entries ({len(metadata_list)}).")
        raise ValueError("Chunks and metadata count mismatch")

    points = []
    for i, chunk in enumerate(chunks):
        point_id = str(uuid.uuid4())
        payload = {"text": chunk.page_content}
        # Merge original chunk metadata with provided metadata_list entry
        if chunk.metadata:
            payload.update(chunk.metadata)
        if metadata_list and metadata_list[i]:
            payload.update(metadata_list[i])

        points.append(models.PointStruct(
            id=point_id,
            vector=embeddings[i],
            payload=payload
        ))

    if not points:
        logger.info(f"No points to add to Qdrant collection '{collection_name}'.")
        return []

    try:
        # Ensure collection exists with the correct vector size (important if size was default and embeddings differ)
        # This uses the global vector_size, make sure it matches embeddings dimension
        create_collection_if_not_exists(collection_name, vector_size=len(embeddings[0]))

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # Wait for operation to complete
        )
        point_ids = [p.id for p in points]
        logger.info(f"Successfully added {len(points)} points to Qdrant collection '{collection_name}'. Point IDs: {point_ids}")
        return point_ids
    except Exception as e:
        logger.error(f"Failed to add documents to Qdrant collection '{collection_name}': {e}")
        # Depending on desired behavior, could re-raise or return None/empty list
        raise
