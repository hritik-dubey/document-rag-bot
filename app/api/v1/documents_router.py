import logging
import shutil
import tempfile
import os
import uuid
from app.core.db import get_async_session
from app.models.db_models import Document as DBDocument # Alias to avoid conflict with langchain.Document
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select # Added select
from qdrant_client import models as qdrant_models # Added qdrant_client.models aliased
from app.core.security import get_current_user, User # User here is AuthUser
from fastapi import APIRouter, UploadFile, File, Path, Depends, HTTPException, status, List # Added List
from app.services import document_processor, vector_store
from app.core.config import settings

logger = logging.getLogger(__name__)
# Assuming LOG_LEVEL is a string like "INFO", "DEBUG"
# BasicConfig should ideally be called once at application startup.
# If main.py's startup event configures logging, this might be redundant or even conflict if not handled carefully.
# For simplicity here, we ensure it's configured if this module is loaded.
# A better approach might be to ensure logger is configured in main and other modules just get the logger.
logging.basicConfig(level=settings.LOG_LEVEL.upper() if hasattr(settings, "LOG_LEVEL") and isinstance(settings.LOG_LEVEL, str) else "INFO")


router = APIRouter()

@router.post("/upload", summary="Document Upload & Processing")
async def upload_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db_session: AsyncSession = Depends(get_async_session)):
    allowed_content_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type. Allowed: {', '.join(allowed_content_types)}")

    # Generate a unique document ID for this upload
    doc_id = str(uuid.uuid4())

        # Create DB Document entry with initial status
        db_doc = DBDocument(
            id=doc_id,
            user_id=current_user.id,  # Assuming current_user from security has an 'id' field for our DB User PK
            filename=file.filename,
            status="uploading",
            qdrant_collection_name=settings.QDRANT_COLLECTION_NAME
        )
        db_session.add(db_doc)
        try:
            await db_session.commit()
            await db_session.refresh(db_doc)
            logger.info(f"Initial DB record created for doc_id: {doc_id}, status: uploading")
        except Exception as e_db_initial:
            await db_session.rollback()
            logger.error(f"Failed to create initial DB record for doc_id {doc_id}: {e_db_initial}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initiate document record in database.")
    temp_file_path = None

    try:
        # Save UploadFile to a temporary file on disk
        # LangChain's TextLoader currently needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Update status to processing
        db_doc.status = "processing"
        db_session.add(db_doc)
        try:
            await db_session.commit()
            await db_session.refresh(db_doc)
            logger.info(f"DB status updated to 'processing' for doc_id: {doc_id}")
        except Exception as e_db_processing:
            # Log error, but might continue processing if DB update fails here, or decide to fail hard
            logger.error(f"Failed to update DB status to 'processing' for doc_id {doc_id}: {e_db_processing}")
            # Not rolling back here as we want to proceed with file processing if possible
            # Consider how to handle this state if DB is crucial for each step

        logger.info(f"User '{current_user.uid}' uploaded file '{file.filename}'. Temp path: {temp_file_path}, Doc ID: {doc_id}")

        # 1. Load document
        docs = document_processor.load_document(temp_file_path, file.content_type)
        if not docs:
            # This case might be covered by load_document raising an error for empty/bad docs,
            # but explicit check here is fine.
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not process the document or document is empty.")

        # 2. Chunk documents
        # These settings could come from config later
        chunks = document_processor.chunk_documents(docs, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to chunk the document.")

        # 3. Generate embeddings
        # This will raise an error if OPENAI_API_KEY is not set
        try:
            embeddings = document_processor.get_document_embeddings(chunks)
        except ValueError as ve: # Catch specific error for missing API key
            logger.error(f"Embedding generation failed for doc_id {doc_id}: {ve}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not generate document embeddings: {ve}. Ensure API key is configured.")
        except Exception as e:
            logger.error(f"Embedding generation failed for doc_id {doc_id}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not generate document embeddings: {e}")

        if not embeddings or len(embeddings) != len(chunks):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate embeddings for all chunks.")

        # 4. Add to Qdrant
        # Prepare metadata for each chunk
        # This metadata will be stored with each vector in Qdrant
        metadata_list = []
        for i, chunk in enumerate(chunks):
            meta = {
                "doc_id": doc_id, # Associate chunk with the uploaded document
                "user_id": current_user.uid,
                "file_name": file.filename,
                "chunk_index": i,
                "original_text_preview": chunk.page_content[:200] # Store a preview
            }
            # If chunk.metadata already has source (from loader), preserve it
            if 'source' in chunk.metadata:
                meta['source_document_path_or_name'] = chunk.metadata['source']
            metadata_list.append(meta)

        try:
            point_ids = vector_store.add_documents_to_qdrant(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                chunks=chunks,
                embeddings=embeddings,
                metadata_list=metadata_list
            )
            if not point_ids:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store document vectors.")
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to Qdrant: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not store document in vector database: {e}")

        logger.info(f"Successfully processed and stored document '{file.filename}' (Doc ID: {doc_id}) for user '{current_user.uid}'. {len(point_ids)} points added to Qdrant.")

        # Update status to completed in DB
        db_doc.status = "completed"
        # db_doc.qdrant_point_ids = point_ids # If we decide to store point_ids as JSON in DB
        db_session.add(db_doc)
        try:
            await db_session.commit()
            await db_session.refresh(db_doc)
            logger.info(f"DB status updated to 'completed' for doc_id: {doc_id}")
        except Exception as e_db_completed:
            logger.error(f"Failed to update DB status to 'completed' for doc_id {doc_id}: {e_db_completed}")
            # Log and continue, as main processing is done. Client gets success, but DB state might be inconsistent.

        # TODO: Store doc_id and other metadata in a relational database (Step 11)

        return {
            "message": "Document uploaded and processed successfully.",
            "doc_id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "qdrant_point_ids": point_ids,
            "content_type": file.content_type
        }

    except ValueError as ve: # Catch specific error for unsupported file type from load_document
        logger.error(f"ValueError during document processing for user '{current_user.uid}', file '{file.filename}': {ve}")
        # Update DB status to failed
        if 'db_doc' in locals() and db_doc: # db_doc should exist if initial DB record was created
            db_doc.status = "failed"
            try:
                db_session.add(db_doc)
                await db_session.commit()
                logger.info(f"DB status updated to 'failed' for doc_id: {db_doc.id} due to unsupported file type.")
            except Exception as e_db_fail:
                logger.error(f"Failed to update DB status to 'failed' for doc_id {db_doc.id}: {e_db_fail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException:
        # Re-raise HTTPExceptions directly
            # Update DB status to failed if db_doc exists
            if 'db_doc' in locals() and db_doc: # Check if db_doc was successfully created
                db_doc.status = "failed"
                try:
                    db_session.add(db_doc)
                    await db_session.commit()
                    logger.info(f"DB status updated to 'failed' for doc_id: {db_doc.id}")
                except Exception as e_db_fail:
                    logger.error(f"Failed to update DB status to 'failed' for doc_id {db_doc.id}: {e_db_fail}")
                    # await db_session.rollback() # Rollback this specific attempt to update status
        raise
    except Exception as e:
        logger.error(f"Error during document upload for user '{current_user.uid}', file '{file.filename}', doc_id '{doc_id}': {e}")
        # Potentially include doc_id in error response if generated
            # Update DB status to failed if db_doc exists
            if 'db_doc' in locals() and db_doc:
                db_doc.status = "failed"
                try:
                    db_session.add(db_doc)
                    await db_session.commit()
                    logger.info(f"DB status updated to 'failed' for doc_id: {db_doc.id}")
                except Exception as e_db_fail:
                    logger.error(f"Failed to update DB status to 'failed' for doc_id {db_doc.id}: {e_db_fail}")
                    # await db_session.rollback() # Rollback this specific attempt to update status
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during document processing: {e}")
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Successfully removed temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_file_path}: {e}")
        if file:
            await file.close()

@router.get("/list", summary="List User Documents", response_model=List[DBDocument])
async def list_documents(current_user: User = Depends(get_current_user), db_session: AsyncSession = Depends(get_async_session)):
    try:
        statement = select(DBDocument).where(DBDocument.user_id == current_user.id)
        results = await db_session.exec(statement)
        documents = results.all()
        logger.info(f"User '{current_user.uid}' listed {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Error listing documents for user '{current_user.uid}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve documents.")

@router.delete("/{document_id}", summary="Delete Document", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str = Path(..., title="Document ID"), current_user: User = Depends(get_current_user), db_session: AsyncSession = Depends(get_async_session)):
    logger.info(f"Attempting to delete document '{document_id}' for user '{current_user.uid}'.")
    try:
        # 1. Fetch the document from DB to ensure it belongs to the user
        statement = select(DBDocument).where(DBDocument.id == document_id, DBDocument.user_id == current_user.id)
        result = await db_session.exec(statement)
        db_doc = result.one_or_none()

        if not db_doc:
            logger.warning(f"Document '{document_id}' not found or does not belong to user '{current_user.uid}'.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found or access denied.")

        # 2. Delete from Qdrant
        qdrant_cli = vector_store.get_qdrant_client()
        if qdrant_cli:
            try:
                logger.info(f"Deleting points from Qdrant for doc_id '{document_id}' in collection '{settings.QDRANT_COLLECTION_NAME}'.")
                # We stored doc_id in the payload of each point.
                qdrant_cli.delete(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(key="doc_id", match=qdrant_models.MatchValue(value=document_id)),
                                # Optionally, also filter by user_id for added security/scoping,
                                # though doc_id should be globally unique and tied to user via DB.
                                qdrant_models.FieldCondition(key="user_id", match=qdrant_models.MatchValue(value=current_user.uid))
                            ]
                        )
                    )
                )
                logger.info(f"Successfully deleted points from Qdrant for doc_id '{document_id}'.")
            except Exception as e_qdrant:
                # Log error but proceed to delete from DB as primary source of truth.
                # Consider implications: orphan data in Qdrant if DB delete fails later? Or inconsistent state.
                # For now, if Qdrant delete fails, we still try to delete from our DB.
                logger.error(f"Failed to delete document '{document_id}' from Qdrant: {e_qdrant}. Proceeding with DB deletion.")
                # Depending on policy, could raise 500 here and stop.
        else:
            logger.warning("Qdrant client not available. Skipping Qdrant deletion for doc_id '{document_id}'.")
            # This could be a critical issue if Qdrant is expected to be up.

        # 3. Delete from relational DB
        await db_session.delete(db_doc)
        await db_session.commit()
        logger.info(f"Successfully deleted document '{document_id}' from database for user '{current_user.uid}'.")

        # HTTP 204 No Content response is returned by FastAPI by default if no content is returned.
        return None # Explicitly return None for 204

    except HTTPException:
        raise # Re-raise HTTPExceptions
    except Exception as e:
        await db_session.rollback() # Rollback DB changes if any other exception occurs
        logger.error(f"Error deleting document '{document_id}' for user '{current_user.uid}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete document: {e}")
