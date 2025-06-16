import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import datetime # Added
import uuid # Added
from app.core.db import get_async_session # Added
from app.models.db_models import ChatSession, Message as DBMessage # Added
from sqlmodel import select # Added
from sqlmodel.ext.asyncio.session import AsyncSession # Added

import json
from app.core.security import get_current_user, User # User is AuthUser
from app.core.config import settings
from fastapi import APIRouter, WebSocket, Depends, HTTPException, status, WebSocketDisconnect, Path # Added Path
from app.services import rag_service

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL.upper() if hasattr(settings, "LOG_LEVEL") and isinstance(settings.LOG_LEVEL, str) else "INFO")

router = APIRouter()

class ChatQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None # Added
    doc_id: Optional[str] = None
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = None # Min score for Qdrant results

@router.post("/query", summary="Standard Chat Query", response_model=Dict[str, Any])
async def query_chat(request: ChatQueryRequest, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_async_session)):
    session_id = request.session_id
    try:
        # 1. Get or Create ChatSession
        if session_id:
            statement = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
            result = await db.exec(statement)
            chat_session = result.one_or_none()
            if not chat_session:
                logger.warning(f"Provided session_id {session_id} not found for user {current_user.uid}. Creating a new session.")
                session_id = None # Force creation of new session

        if not session_id:
            new_session_id = str(uuid.uuid4())
            chat_session = ChatSession(id=new_session_id, user_id=current_user.id, title=request.query[:50]) # Use query start as title
            db.add(chat_session)
            await db.commit()
            await db.refresh(chat_session)
            session_id = chat_session.id
            logger.info(f"New chat session created with id: {session_id} for user {current_user.uid}")
        else:
            # Update last_active time for existing session
            chat_session.last_active = datetime.datetime.utcnow()
            db.add(chat_session)
            await db.commit()
            await db.refresh(chat_session)
            logger.info(f"Using existing chat session: {session_id} for user {current_user.uid}")

        # 2. Store User Message
        user_message = DBMessage(session_id=session_id, role="user", content=request.query)
        db.add(user_message)

        # 3. Call RAG Pipeline
        rag_result = rag_service.query_rag_pipeline(
            user_id=current_user.uid,
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k or 5, # Ensure top_k has a default
            score_threshold=request.score_threshold
        )

        # 4. Store AI Message
        ai_message_content = rag_result.get("answer", "Error: No answer from RAG pipeline")
        ai_message = DBMessage(session_id=session_id, role="ai", content=ai_message_content)
        db.add(ai_message)

        await db.commit() # Commit user and AI messages together
        logger.info(f"User and AI messages stored for session {session_id}")

        # Add session_id to the response
        rag_result["session_id"] = session_id
        return rag_result

    except ValueError as ve:
        logger.error(f"ValueError in chat query for user {current_user.uid}, session {session_id}: {ve}")
        try: await db.commit()
        except: pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ve))
    except Exception as e:
        logger.error(f"Unhandled error in chat query for user {current_user.uid}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while processing your chat query.")

@router.websocket("/stream")
async def stream_chat(websocket: WebSocket, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_async_session)):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for user: {current_user.uid}")

    chat_session_for_query: Optional[ChatSession] = None # To store/reuse session within this connection if client doesn't manage session_id

    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            query = data.get("query")
            client_session_id = data.get("session_id")
            doc_id = data.get("doc_id")
            top_k = data.get("top_k", 5)
            score_threshold = data.get("score_threshold") # Added

            if not query:
                await websocket.send_json({"type": "error", "content": "Query is missing."})
                continue

            current_query_session_id = None
            if client_session_id: # Client explicitly provides a session_id
                stmt = select(ChatSession).where(ChatSession.id == client_session_id, ChatSession.user_id == current_user.id)
                result = await db.exec(stmt)
                chat_session_for_query = result.one_or_none()
                if chat_session_for_query:
                    chat_session_for_query.last_active = datetime.datetime.utcnow()
                    db.add(chat_session_for_query)
                    current_query_session_id = chat_session_for_query.id
                    logger.info(f"Using existing session {current_query_session_id} for WS query from user {current_user.uid}")
                else:
                    logger.warning(f"Client provided session_id {client_session_id} not found/owned by user {current_user.uid}. Creating new session.")

            if not current_query_session_id: # No valid client_session_id, or it was invalid
                new_session_id_val = str(uuid.uuid4())
                chat_session_for_query = ChatSession(id=new_session_id_val, user_id=current_user.id, title=query[:50])
                db.add(chat_session_for_query)
                current_query_session_id = new_session_id_val
                logger.info(f"New session {current_query_session_id} created for WS query from user {current_user.uid}")

            await websocket.send_json({"type": "session_info", "session_id": current_query_session_id})

            user_message = DBMessage(session_id=current_query_session_id, role="user", content=query)
            db.add(user_message)
            try:
                await db.commit()
                if chat_session_for_query: await db.refresh(chat_session_for_query)
                await db.refresh(user_message)
                logger.info(f"Stored user message for session {current_query_session_id}")
            except Exception as e_db_user_msg:
                logger.error(f"DB error storing user message/session for {current_query_session_id}: {e_db_user_msg}")
                await db.rollback()
                await websocket.send_json({"type": "error", "content": "Database error processing your request."})
                continue

            full_ai_response = []
            async for chunk in rag_service.stream_rag_pipeline(
                user_id=current_user.uid, query=query, doc_id=doc_id, top_k=top_k, score_threshold=score_threshold
            ):
                await websocket.send_json(chunk)
                if chunk.get("type") == "token":
                    full_ai_response.append(chunk.get("content", ""))
                elif chunk.get("type") == "sources":
                    logger.debug(f"Sources received for session {current_query_session_id}: {chunk.get('content')}")
                elif chunk.get("type") == "error":
                    if not full_ai_response: full_ai_response.append(f"Error: {chunk.get('content')}")

                if chunk.get("type") == "end":
                    break

            if full_ai_response:
                ai_content = "".join(full_ai_response)
                ai_message_record = DBMessage(session_id=current_query_session_id, role="ai", content=ai_content)
                db.add(ai_message_record)
                try:
                    await db.commit()
                    await db.refresh(ai_message_record) # Refresh to get timestamp etc.
                    logger.info(f"Stored AI message for session {current_query_session_id}")
                except Exception as e_db_ai_msg:
                    logger.error(f"DB error storing AI message for session {current_query_session_id}: {e_db_ai_msg}")
                    await db.rollback()
            else:
                logger.warning(f"No AI response content to store for session {current_query_session_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {current_user.uid}")
    except json.JSONDecodeError:
        logger.warning(f"WebSocket received non-JSON message from {current_user.uid}. Attempting to close gracefully if possible.")
        try: await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        except: pass # already closed
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket for user {current_user.uid}: {e}")
        try: await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except: pass
    finally:
        logger.info(f"Closing WebSocket connection handler for user: {current_user.uid if current_user else 'Unknown'}")

@router.get("/history/{session_id}", summary="Get Chat History for a Session", response_model=List[DBMessage])
async def get_chat_history(
    session_id: str = Path(..., title="Session ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    logger.info(f"Fetching chat history for session '{session_id}' for user '{current_user.uid}'.")
    try:
        # First, verify the session belongs to the user
        session_check_stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        session_result = await db.exec(session_check_stmt)
        session = session_result.one_or_none()

        if not session:
            logger.warning(f"Chat session '{session_id}' not found or does not belong to user '{current_user.uid}'.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied.")

        # Fetch messages for the session, ordered by timestamp
        messages_stmt = select(DBMessage).where(DBMessage.session_id == session_id).order_by(DBMessage.timestamp)
        messages_result = await db.exec(messages_stmt)
        messages = messages_result.all()

        logger.info(f"Retrieved {len(messages)} messages for session '{session_id}'.")
        return messages

    except HTTPException:
        raise # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error fetching chat history for session '{session_id}', user '{current_user.uid}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chat history.")
