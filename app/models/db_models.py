from sqlmodel import SQLModel, Field, Relationship, Column, DateTime, String
from typing import Optional, List
import datetime
import uuid # For default UUIDs

class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, index=True)
    firebase_uid: str = Field(sa_column=Column(String, unique=True, index=True, nullable=False)) # Firebase UID
    email: Optional[str] = Field(default=None, index=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    subscription_tier: Optional[str] = Field(default="free")

    documents: List["Document"] = Relationship(back_populates="owner")
    chat_sessions: List["ChatSession"] = Relationship(back_populates="user")

class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, index=True) # This is our internal doc_id
    user_id: str = Field(foreign_key="user.id", index=True)
    filename: str
    upload_date: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    status: str = Field(default="processing") # e.g., processing, completed, failed
    qdrant_collection_name: Optional[str] = Field(default=None) # In case we use per-doc collections or specific names
    # qdrant_point_ids: List[str] = Field(default=[], sa_column=Column(JSON)) # If storing point IDs, might be complex

    owner: Optional[User] = Relationship(back_populates="documents")

class ChatSession(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, index=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    last_active: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    # Potentially a title or summary for the session
    title: Optional[str] = Field(default=None)

    user: Optional[User] = Relationship(back_populates="chat_sessions")
    messages: List["Message"] = Relationship(back_populates="session")

class Message(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, index=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    role: str # e.g., "user", "ai", "system"
    content: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    # Could add reaction, citations (JSON string or relationship to DocumentChunk table if very granular)
    # metadata: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSONB))

    session: Optional[ChatSession] = Relationship(back_populates="messages")
