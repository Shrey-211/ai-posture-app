from pydantic import BaseModel, EmailStr

class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    code: str

class GoogleLogin(BaseModel):
    id_token: str
