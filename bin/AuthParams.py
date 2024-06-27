from pydantic import BaseModel

class AuthParams(BaseModel):
    username: str
    password: str
