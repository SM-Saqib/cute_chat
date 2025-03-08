from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse

from sqlalchemy.future import select
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List

from app.services import OpenAIClient
from app.db.base import get_db, get_async_db
from app.model import Conversation
from app.services import RedisCache

router = APIRouter()

openai_client = OpenAIClient()

redis_cache = RedisCache(host="localhost", port=6379, db=0)


@router.post("/chat")
async def chat(message: str, user_id: int, db: Session = Depends(get_async_db)):
    cache_key = f"{user_id}:{message}"
    cached_response = redis_cache.get_json(cache_key)

    if cached_response:
        return JSONResponse(content={"status": "success", "ai_reply": cached_response})

    async with get_async_db() as db:
        conversation = Conversation(
            message=message,
            ai_reply="",
            timestamp=datetime.now(),
            user_id=user_id,
        )
        db.add(conversation)
        await db.commit()

        history_cache_key = f"{user_id}:history"
        history = redis_cache.get_json(history_cache_key)

        if not history:
            result = await db.execute(
                select(Conversation)
                .filter_by(user_id=conversation.user_id)
                .order_by(Conversation.created_at.desc())
                .limit(20)
            )
            history = result.scalars().all()
            redis_cache.set_json(
                history_cache_key,
                [h.message for h in history],
                ex=timedelta(minutes=60),
            )
        else:
            history = [Conversation(message=h) for h in history]

    openai_client.chat_history[conversation.user_phone_number] = [
        {"role": "user", "content": h.message} for h in history
    ]

    response = await openai_client.get_completion_from_messages_with_context(
        messages=[{"role": "user", "content": message}],
        phone_no=conversation.user_phone_number,
    )

    conversation.ai_reply = response
    await db.commit()

    redis_cache.set_json(cache_key, response, ex=timedelta(minutes=60))

    return JSONResponse(content={"status": "success", "ai_reply": response})
