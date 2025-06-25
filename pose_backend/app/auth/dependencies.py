from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from app.auth.auth_utils import decode_access_token
from app.auth.models import User
from sqlmodel import Session, select
from app.database import get_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    payload = decode_access_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
