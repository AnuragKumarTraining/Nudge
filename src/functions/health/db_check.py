from sqlalchemy import text
from sqlalchemy.orm import Session
from src.database.connection import engine
from src.database.session import get_db

def check_db_connection() -> dict:
    """Tests the database connection via direct engine connection."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            if result == 1:
                return {"status": "ok", "database": "connected"}
            return {"status": "error", "database": "unexpected response"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_db_session() -> dict:
    """Tests the database connection using the session generator (get_db)."""
    db_gen = get_db()
    db: Session = next(db_gen)
    try:
        result = db.execute(text("SELECT 1")).scalar()
        if result == 1:
            return {"status": "ok", "session": "healthy"}
        return {"status": "error", "session": "unexpected response"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass