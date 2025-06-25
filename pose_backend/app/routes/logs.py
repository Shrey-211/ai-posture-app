from fastapi import APIRouter, Depends
from sqlmodel import Session
from app.database import get_session
from app.schemas.pose_log import PoseLogCreate
from app.models.pose_log import PoseLog
from app.auth.dependencies import get_current_user
from app.auth.models import User

router = APIRouter()

@router.post("/log_pose_event")
def log_pose_event(
    data: PoseLogCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    log = PoseLog(**data.dict())
    session.add(log)
    session.commit()
    return {"status": "success"}
