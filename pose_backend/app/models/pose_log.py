from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class PoseLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    device_id: str
    timestamp: datetime
    exercise: str
    event: str
    angle: float
    duration: float
