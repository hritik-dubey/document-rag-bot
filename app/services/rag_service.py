from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import asyncio
from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import models # For Filter

from app.services import vector_store
from app.services.document_processor import get_openai_embeddings
from app.core.config import settings
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL.upper())

chat_model = None

def get_chat_model():
    global chat_model
    if not chat_model:
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not configured. Cannot initialize ChatOpenAI.")
            raise ValueError("OPENAI_API_KEY not set for Chat Model")
        try:
            chat_model = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, temperature=0.3) # Model can be configured more
            logger.info("ChatOpenAI model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise
    return chat_model

DEFAULT_FALLBACK_RESPONSE = "I couldn't find an answer in the provided documents to your question. Please try rephrasing or ask about a different topic based on the uploaded content."
MAX_CONTEXT_CHAR_LENGTH = 8000 # Define max context length

def query_rag_pipeline(user_id: str, query: str, doc_id: str | None = None, top_k: int = 5, score_threshold: float | None = None) -> Dict[str, Any]:
    logger.info(f"RAG query for user '{user_id}', query: '{query}', doc_id: {doc_id}, top_k: {top_k}, score_threshold: {score_threshold}")
    embeddings_model = get_openai_embeddings()
    qdrant = vector_store.get_qdrant_client()

    if not qdrant:
        logger.error("Qdrant client not available for RAG query.")
        return {"answer": "Error: Vector database is not available.", "sources": []}

    # 1. Embed the query
    try:
        query_embedding = embeddings_model.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query for user '{user_id}': {e}")
        return {"answer": "Error: Could not process your query embedding.", "sources": []}

    # 2. Construct Qdrant filter
    query_filter = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
    if doc_id:
        query_filter.must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        logger.info(f"Applying filter for user_id: {user_id} AND doc_id: {doc_id}")
    else:
        logger.info(f"Applying filter for user_id: {user_id} (all documents)")

    # 3. Search Qdrant for relevant chunks
    try:
        search_results = qdrant.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )
        logger.debug(f"Qdrant search for user '{user_id}' returned {len(search_results)} results.")
    except Exception as e:
        logger.error(f"Qdrant search failed for user '{user_id}': {e}")
        return {"answer": "Error: Failed to retrieve relevant documents from vector database.", "sources": []}

    if not search_results:
        logger.warning(f"No relevant chunks found in Qdrant for user '{user_id}' query: '{query}' with filter: {query_filter}")
        return {"answer": DEFAULT_FALLBACK_RESPONSE, "sources": []}

    # 4. Format context and extract source information
    context_parts = []
    sources_for_response = []
    for hit in search_results:
        if hit.payload:
            context_parts.append(hit.payload.get("text", ""))
            source_info = {
                "doc_id": hit.payload.get("doc_id"),
                "file_name": hit.payload.get("file_name"),
                "chunk_index": hit.payload.get("chunk_index"),
                "score": hit.score,
                "preview": hit.payload.get("text","")[:100] + "..." # Short preview
            }
            sources_for_response.append(source_info)
    context_str = "\n\n---\n\n".join(filter(None, context_parts))

    if len(context_str) > MAX_CONTEXT_CHAR_LENGTH:
        logger.warning(f"Context length ({len(context_str)} chars) exceeds max ({MAX_CONTEXT_CHAR_LENGTH} chars). Truncating for user '{user_id}'.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LENGTH]

    if not context_str.strip():
        logger.warning(f"Context string is empty after processing Qdrant results for user '{user_id}' query: '{query}'")
        return {"answer": DEFAULT_FALLBACK_RESPONSE, "sources": []}

    # 5. Construct RAG chain and invoke
    llm = get_chat_model()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based ONLY on the following context. If the context does not contain the answer, say so. Do not use any prior knowledge.\n\nContext:\n{context}"),
        ("human", "Question: {question}")
    ])

    # Using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": lambda x: context_str, "question": RunnablePassthrough()} # Pass query directly as 'question'
        | prompt_template
        | llm
        | StrOutputParser()
    )

    try:
        answer = rag_chain.invoke(query) # Pass the original query string
        logger.info(f"RAG chain invoked successfully for user '{user_id}'. Answer: '{answer[:100]}...'")
    except Exception as e:
        logger.error(f"RAG chain invocation failed for user '{user_id}': {e}")
        return {"answer": "Error: Failed to generate an answer using the language model.", "sources": sources_for_response} # Return sources even if LLM fails

    return {"answer": answer, "sources": sources_for_response}


async def stream_rag_pipeline(user_id: str, query: str, doc_id: str | None = None, top_k: int = 5, score_threshold: float | None = None) -> AsyncGenerator[Dict[str, Any], None]:
    logger.info(f"Streaming RAG query for user '{user_id}', query: '{query}', doc_id: {doc_id}, top_k: {top_k}, score_threshold: {score_threshold}")
    embeddings_model = get_openai_embeddings()
    qdrant = vector_store.get_qdrant_client()

    if not qdrant:
        logger.error("Qdrant client not available for streaming RAG query.")
        yield {"type": "error", "content": "Error: Vector database is not available."}
        return

    # 1. Embed the query (non-streaming part)
    try:
        query_embedding = embeddings_model.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query for user '{user_id}': {e}")
        yield {"type": "error", "content": "Error: Could not process your query embedding."}
        return

    # 2. Construct Qdrant filter (non-streaming part)
    query_filter = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
    if doc_id:
        query_filter.must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))

    # 3. Search Qdrant (non-streaming part)
    try:
        search_results = qdrant.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )
    except Exception as e:
        logger.error(f"Qdrant search failed for user '{user_id}': {e}")
        yield {"type": "error", "content": "Error: Failed to retrieve relevant documents."}
        return

    if not search_results:
        logger.warning(f"No relevant chunks found in Qdrant for user '{user_id}' (streaming) query: '{query}'")
        yield {"type": "info", "content": DEFAULT_FALLBACK_RESPONSE}
        # Optionally send sources if you want to show "no results from these documents"
        # yield {"type": "sources", "content": []}
        yield {"type": "end"}
        return

    # 4. Format context and extract source information (non-streaming part)
    context_parts = []
    sources_for_response = []
    for hit in search_results:
        if hit.payload:
            context_parts.append(hit.payload.get("text", ""))
            sources_for_response.append({
                "doc_id": hit.payload.get("doc_id"),
                "file_name": hit.payload.get("file_name"),
                "chunk_index": hit.payload.get("chunk_index"),
                "score": hit.score,
                "preview": hit.payload.get("text","")[:100] + "..."
            })
    context_str = "\n\n---\n\n".join(filter(None, context_parts))

    # Send sources first
    yield {"type": "sources", "content": sources_for_response}

    if len(context_str) > MAX_CONTEXT_CHAR_LENGTH:
        logger.warning(f"Context length ({len(context_str)} chars) exceeds max ({MAX_CONTEXT_CHAR_LENGTH} chars). Truncating for user '{user_id}'.")
        context_str = context_str[:MAX_CONTEXT_CHAR_LENGTH]

    if not context_str.strip():
        logger.warning(f"Context string is empty after processing Qdrant results for user '{user_id}' (streaming) query: '{query}'")
        yield {"type": "info", "content": DEFAULT_FALLBACK_RESPONSE}
        yield {"type": "end"}
        return

    # 5. Construct RAG chain for streaming
    stream_handler = AsyncIteratorCallbackHandler()
    # llm = get_chat_model() # Assuming get_chat_model() can be used as is
    # Update get_chat_model to accept streaming=True and callbacks if needed, or create a new one.
    # For now, let's try to configure it directly if ChatOpenAI supports it easily.
    # ChatOpenAI needs `streaming=True` and `callbacks=[stream_handler]` for this to work.

    # Re-initialize chat_model for streaming if necessary or modify get_chat_model
    # This is a simplification; ideally, get_chat_model would take a streaming flag.
    try:
        streaming_llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            streaming=True,
            callbacks=[stream_handler]
        )
    except Exception as e:
        logger.error(f"Failed to initialize Streaming ChatOpenAI: {e}")
        yield {"type": "error", "content": "Error: Could not initialize streaming language model."}
        return

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based ONLY on the following context. If the context does not contain the answer, say so. Do not use any prior knowledge. Stream your answer token by token.\n\nContext:\n{context}"),
        ("human", "Question: {question}")
    ])

    rag_chain = (
        {"context": lambda x: context_str, "question": RunnablePassthrough()}
        | prompt_template
        | streaming_llm # Use the streaming_llm instance
        | StrOutputParser()
    )

    # Start LLM generation in a separate task so we can yield tokens as they arrive
    async def run_llm_chain():
        try:
            await rag_chain.ainvoke(query)
        except Exception as e:
            logger.error(f"Streaming RAG chain invocation failed for user '{user_id}': {e}")
            # Try to send an error token through the handler if possible, or log
            # This error might occur after some tokens have already been sent.
            # The handler might close itself, or we might need to yield an error here.
            # For now, logging it. The client might see an abrupt end of stream.
            try:
                await stream_handler.on_llm_error(e) # Notify handler
            except Exception as e_cb:
                logger.error(f"Error notifying callback handler of LLM error: {e_cb}")

    # Run the LLM chain in the background
    asyncio.create_task(run_llm_chain())

    # Yield tokens from the handler
    try:
        async for token in stream_handler.aiter():
            yield {"type": "token", "content": token}
    except Exception as e:
        logger.error(f"Error iterating over stream handler for user '{user_id}': {e}")
        yield {"type": "error", "content": "Error during response streaming."}
    finally:
        logger.info(f"Finished streaming RAG response for user '{user_id}', query: '{query}'")
        yield {"type": "end"} # Signal end of stream


# need to implement