from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from datetime import datetime, timedelta
from app.auth.models import OTPCode, User
from app.auth.schemas import OTPRequest, OTPVerify
from app.auth.auth_utils import generate_otp, create_access_token
from app.database import get_session

router = APIRouter()

@router.post("/send_otp")
def send_otp(request: OTPRequest, session: Session = Depends(get_session)):
    otp = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=5)
    otp_entry = OTPCode(email=request.email, code=otp, expires_at=expires_at)
    session.add(otp_entry)
    session.commit()
    print(f"OTP for {request.email} is: {otp}")
    return {"message": "OTP sent"}

@router.post("/verify_otp")
def verify_otp(request: OTPVerify, session: Session = Depends(get_session)):
    record = session.exec(
        select(OTPCode)
        .where(OTPCode.email == request.email)
        .where(OTPCode.code == request.code)
        .where(OTPCode.expires_at > datetime.utcnow())
        .where(OTPCode.verified == False)
    ).first()
    if not record:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")

    record.verified = True
    session.add(record)

    user = session.exec(select(User).where(User.email == request.email)).first()
    if not user:
        user = User(email=request.email)
        session.add(user)
        session.commit()
        session.refresh(user)

    token = create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}
