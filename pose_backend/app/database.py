from sqlmodel import SQLModel, create_engine, Session
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pose_logs.db")
engine = create_engine(DATABASE_URL, echo=True)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    from app.auth.models import User, OTPCode
    from app.models.pose_log import PoseLog
    SQLModel.metadata.create_all(engine)
