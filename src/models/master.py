from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


class Master(Base):
    __tablename__ = "masters"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    property_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("properties.id"),
        nullable=False,
    )

    room_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("rooms.id"),
        nullable=False,
    )

    asset_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("assets.id"),
        nullable=False,
    )

    master_image_url: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    prompt_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("prompts.id"),
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

    room: Mapped["Room"] = relationship(
        "Room",
        back_populates="masters",
    )

    asset: Mapped["Asset"] = relationship(
        "Asset",
        back_populates="masters",
    )

    prompt: Mapped["Prompt | None"] = relationship(
        "Prompt",
        back_populates="masters",
    )