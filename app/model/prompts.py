from pydantic import BaseModel
from app.db.base import Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship


class Prompts(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    prompt = Column(String(100))
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now())

    def __repr__(self):
        return f"Prompts(id={self.id!r}, prompt={self.prompt!r}, user_id={self.user_id!r}, created_at={self.created_at!r})"
