from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from app.services import OpenAIClient
from app.db.base import get_db
from app.model import Prompts
from sqlalchemy.future import select
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List

router = APIRouter()

openai_client = OpenAIClient()


@router.post("/prompts/create")
async def create_prompt(prompt: str, db: Session = Depends(get_db)):
    """simply store the prompt into the database"""

    prompt_db = Prompts(prompt=prompt)
    db.add(prompt_db)
    await db.commit()

    return JSONResponse(content={"status": "success"})


@router.get("/prompts/get")
async def get_prompts(db: Session = Depends(get_db)):
    result = await db.execute(select(Prompts))
    prompts = result.scalars().all()
    return JSONResponse(content={"status": "success", "prompts": prompts})


@router.delete("/prompts/delete/{prompt_id}")
async def delete_prompt(prompt_id: int, db: Session = Depends(get_db)):
    result = await db.execute(select(Prompts).filter_by(id=prompt_id))
    prompt = result.scalars().one()
    db.delete(prompt)
    await db.commit()
    return JSONResponse(content={"status": "success"})


@router.put("/prompts/update/{prompt_id}")
async def update_prompt(prompt_id: int, prompt: str, db: Session = Depends(get_db)):
    result = await db.execute(select(Prompts).filter_by(id=prompt_id))
    prompt_db = result.scalars().one()
    prompt_db.prompt = prompt
    await db.commit()
    return JSONResponse(content={"status": "success"})
