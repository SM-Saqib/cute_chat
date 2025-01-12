from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from app.services import OpenAIClient
from app.db.base import get_db
from app.model import Conversation
from sqlalchemy.future import select
from datetime import datetime, timedelta
from typing import List

router = APIRouter()

openai_client = OpenAIClient()

@router.post("/chat")
async def chat(message: str, request: Request):
    db = get_db()
    conversation = Conversation(
        message=message,
        ai_reply="",
        timestamp=datetime.now(),
    )
    db.add(conversation)
    await db.commit()

    result = await db.execute(
        select(Conversation).filter_by(user_id=conversation.user_id).order_by(Conversation.created_at.desc()).limit(20)
    )
    history = result.scalars().all()

    openai_client.chat_history[conversation.user_phone_number] = [
        {"role": "user", "content": h.message} for h in history
    ]

    response = await openai_client.get_completion_from_messages_with_context(
        messages=[{"role": "user", "content": message}],
        phone_no=conversation.user_phone_number,
        db=db,
    )

    conversation.ai_reply = response
    await db.commit()

    return JSONResponse(content={"status": "success", "ai_reply": response})
