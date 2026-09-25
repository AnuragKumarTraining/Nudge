from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


class Room(Base):
    __tablename__ = "rooms"

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

    room_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    property: Mapped["Property"] = relationship(
        "Property",
        back_populates="rooms",
    )

    assets: Mapped[list["Asset"]] = relationship(
        "Asset",
        back_populates="room",
    )

    masters: Mapped[list["Master"]] = relationship(
        "Master",
        back_populates="room",
    )