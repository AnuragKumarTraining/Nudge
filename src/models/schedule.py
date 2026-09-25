from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.enums import RecurrenceType, ScheduleType


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    property_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("properties.id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    schedule_type: Mapped[ScheduleType] = mapped_column(
        Enum(ScheduleType),
        nullable=False,
    )

    recurrence_type: Mapped[RecurrenceType | None] = mapped_column(
        Enum(RecurrenceType),
        nullable=True,
    )

    custom_days: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    start_time_minutes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    grace_period_minutes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    property: Mapped["Property"] = relationship(
        "Property",
        back_populates="schedules",
    )

    captures: Mapped[list["Capture"]] = relationship(
        "Capture",
        back_populates="schedule",
    )