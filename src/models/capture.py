from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.enums import CaptureStatus


class Capture(Base):
    __tablename__ = "captures"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    worker_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id"),
        nullable=False,
    )

    asset_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("assets.id"),
        nullable=False,
    )

    schedule_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("schedules.id"),
        nullable=True,
    )

    capture_image_url: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    status: Mapped[CaptureStatus] = mapped_column(
        Enum(CaptureStatus),
        nullable=False,
    )

    issue_description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    capture_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
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

    # Capture → Worker
    worker: Mapped["User"] = relationship(
        "User",
        foreign_keys=[worker_id],
        back_populates="captures",
    )

    # Capture → Asset
    asset: Mapped["Asset"] = relationship(
        "Asset",
        back_populates="captures",
    )

    # Capture → Schedule
    schedule: Mapped["Schedule | None"] = relationship(
        "Schedule",
        back_populates="captures",
    )