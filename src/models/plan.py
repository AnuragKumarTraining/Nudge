from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    max_rooms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    max_workers: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    ai_analysis_frequency: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    price_monthly: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    properties: Mapped[list["Property"]] = relationship(
        "Property",
        back_populates="plan",
    )