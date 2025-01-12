# saving all user /ai messages in this db

from pydantic import BaseModel
from app.db.base import Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship


class Conversation(Base):
    __tablename__ = "conversation"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, index=True)
    ai_reply = Column(String, index=True)
    user_id = Column(Integer, index=True)
    created_at = Column(DateTime, index=True)
