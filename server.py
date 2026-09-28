from fastapi import FastAPI
from src.functions.health.check import router as health_router

app = FastAPI()

app.include_router(health_router)