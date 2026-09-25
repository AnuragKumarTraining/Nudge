from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.enums import (
    AuthProvider,
    InviteMethod,
    PropertyWorkerStatus,
    Role,
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    email: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
    )

    password_hash: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    auth_provider: Mapped[AuthProvider] = mapped_column(
        Enum(AuthProvider),
        nullable=False,
    )

    phone: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    is_phone_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    first_name: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    last_name: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    role: Mapped[Role] = mapped_column(
        Enum(Role),
        nullable=False,
    )

    assigned_property_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey(
            "properties.id",
            use_alter=True,
            name="fk_users_assigned_property",
        ),
        nullable=True,
    )

    worker_status: Mapped[PropertyWorkerStatus | None] = mapped_column(
        Enum(PropertyWorkerStatus),
        nullable=True,
    )

    invited_via: Mapped[InviteMethod | None] = mapped_column(
        Enum(InviteMethod),
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

    # User → Sessions
    sessions: Mapped[list["Session"]] = relationship(
        "Session",
        back_populates="user",
    )

    # User → Properties they own
    owned_properties: Mapped[list["Property"]] = relationship(
        "Property",
        foreign_keys="Property.owner_id",
        back_populates="owner",
    )

    # User → Property they are assigned to as a worker
    assigned_property: Mapped["Property | None"] = relationship(
        "Property",
        foreign_keys=[assigned_property_id],
        back_populates="assigned_workers",
    )

    # User → Captures created as a worker
    captures: Mapped[list["Capture"]] = relationship(
        "Capture",
        foreign_keys="Capture.worker_id",
        back_populates="worker",
    )