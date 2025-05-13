from pydantic import BaseModel, constr,  field_validator
import re
from typing import Literal


PW_REGEX = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$")

class UserCreate(BaseModel):
    username: str
    password: constr(min_length=8)
    role: Literal["user", "authority", "admin"] = "user"


    @field_validator("password")
    def strong_password(cls, v) -> str:
        if not PW_REGEX.fullmatch(v):
            raise ValueError(
                "Password must be ≥8 chars and contain letters & digits"
            )
        return v

class UserOut(BaseModel):
    username: str
    role: str

class Token(BaseModel):
    access_token: str
    refresh_token: str 
    token_type: str

class RefreshIn(BaseModel):
    refresh_token: str

