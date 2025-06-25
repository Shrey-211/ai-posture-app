from fastapi import FastAPI
from app.database import init_db
from app.auth import routes_otp, routes_google
from app.routes import logs

app = FastAPI(title="Pose Detection Auth Backend")

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(routes_otp.router)
app.include_router(routes_google.router)
app.include_router(logs.router)
