import os
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor, register_uuid
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()
register_uuid()          # return uuid columns as uuid.UUID objects
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 5, dsn=os.environ["DATABASE_URL"])
    return _pool

@contextmanager
def transaction():
    """Yield a dict cursor; commit on success, roll back on any error."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)