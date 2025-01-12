from pydantic import BaseModel

class UserCreate(BaseModel):
    phone_number: str
    first_name: str
    last_name: str
    password: str