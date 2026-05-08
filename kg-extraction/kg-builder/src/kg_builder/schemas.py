from pydantic import BaseModel, field_validator


class Triple(BaseModel):
    h: str
    r: str
    t: str

    @field_validator('h', 'r', 't')
    @classmethod
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError('Field cannot be empty or whitespace only')
        return v


class Document(BaseModel):
    id: str
    title: str = ""
    abstract: str = ""
    body: str = ""


# Config models are in config.py
