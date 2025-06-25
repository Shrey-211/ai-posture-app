from pydantic import BaseModel
from datetime import datetime

class PoseLogCreate(BaseModel):
    device_id: str
    timestamp: datetime
    exercise: str
    event: str
    angle: float
    duration: float
