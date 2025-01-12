from pydantic import BaseModel

class Prompts(BaseModel):
    prompt: str
    category: str = "general"

