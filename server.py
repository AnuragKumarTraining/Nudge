from fastapi import FastAPI
from src.functions.health.check import router as health_router
from src.functions.login.route import router as login_router

app = FastAPI()

app.include_router(health_router)
app.include_router(login_router)