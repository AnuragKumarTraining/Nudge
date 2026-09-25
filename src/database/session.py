from sqlalchemy.orm import Session

from src.database.connection import engine


def get_db():
    db = Session(engine)

    try:
        yield db
    finally:
        db.close()