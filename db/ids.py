import uuid

def new_id() -> uuid.UUID:
    """UUIDv4 for tenants, properties, rooms, masters."""
    return uuid.uuid4()

try:
    new_time_id = uuid.uuid7          # Python 3.14+
except AttributeError:
    from uuid6 import uuid7 as new_time_id   # pip install uuid6