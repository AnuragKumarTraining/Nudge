from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.enums import AssetCategory


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    room_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("rooms.id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    category: Mapped[AssetCategory] = mapped_column(
        Enum(AssetCategory),
        nullable=False,
    )

    room: Mapped["Room"] = relationship(
        "Room",
        back_populates="assets",
    )

    masters: Mapped[list["Master"]] = relationship(
        "Master",
        back_populates="asset",
    )

    captures: Mapped[list["Capture"]] = relationship(
        "Capture",
        back_populates="asset",
    )