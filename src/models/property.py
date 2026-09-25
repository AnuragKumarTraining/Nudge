from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.enums import PropertyType, VerificationStatus


class Property(Base):
    __tablename__ = "properties"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    owner_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    address: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    timezone: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    property_type: Mapped[PropertyType] = mapped_column(
        Enum(PropertyType),
        nullable=False,
    )

    plan_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("plans.id"),
        nullable=False,
    )

    verification_status: Mapped[VerificationStatus] = mapped_column(
        Enum(VerificationStatus),
        nullable=False,
    )

    verification_doc_url: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Property → Owner
    owner: Mapped["User"] = relationship(
        "User",
        foreign_keys=[owner_id],
        back_populates="owned_properties",
    )

    # Property → Plan
    plan: Mapped["Plan"] = relationship(
        "Plan",
        back_populates="properties",
    )

    # Property → Workers
    assigned_workers: Mapped[list["User"]] = relationship(
        "User",
        foreign_keys="User.assigned_property_id",
        back_populates="assigned_property",
    )

    # Property → Rooms
    rooms: Mapped[list["Room"]] = relationship(
        "Room",
        back_populates="property",
    )

    # Property → Schedules
    schedules: Mapped[list["Schedule"]] = relationship(
        "Schedule",
        back_populates="property",
    )