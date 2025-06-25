from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.auth.schemas import GoogleLogin
from app.auth.models import User
from app.auth.auth_utils import create_access_token
from app.config import GOOGLE_CLIENT_ID
from app.database import get_session
import requests

router = APIRouter()

@router.post("/google_login")
def google_login(request: GoogleLogin, session: Session = Depends(get_session)):
    res = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={request.id_token}")
    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    data = res.json()
    if data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Client ID mismatch")

    email = data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")

    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(email=email)
        session.add(user)
        session.commit()
        session.refresh(user)

    token = create_access_token(data={"sub": email})
    return {"access_token": token, "token_type": "bearer"}
