from fastapi import APIRouter, Depends, HTTPException, status
from src.database.session import get_db
from src.functions.health.db_check import check_db_connection

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/db")
def health_check():
    result = check_db_connection()
    if result.get("status") != "ok":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=result
        )
    return result