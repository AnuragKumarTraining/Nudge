from sqlalchemy import inspect

from src.database.base import Base

# Import all models so SQLAlchemy registers them
from src.models import (
    User,
    Session,
    Plan,
    Property,
    Room,
    Asset,
    Master,
    Schedule,
    Capture,
    Prompt,
)


print("Models loaded successfully!")
print()

for table_name, table in Base.metadata.tables.items():
    print(f"Table: {table_name}")

    for column in table.columns:
        print(
            f"  - {column.name}: "
            f"{column.type} "
            f"{'PK' if column.primary_key else ''}"
        )

    print()