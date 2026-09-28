import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Boolean, Float, DateTime, ForeignKey, Enum as SQLEnum, Text
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# Import enums from your enums.py file
from src.database.enums import (
    Role, CaptureStatus, VerificationStatus, ScheduleType, 
    RecurrenceType, SpaceType, PropertyWorkerStatus, IssueStatus, IssueSeverity
)

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class AppUser(Base):
    __tablename__ = 'app_users'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    phone = Column(String(50))
    is_phone_verified = Column(Boolean, default=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    sessions = relationship("UserSession", back_populates="user")
    owned_properties = relationship("Property", back_populates="owner")
    worker_profiles = relationship("PropertyWorker", back_populates="user")
    assigned_issues = relationship("IssueTicket", foreign_keys='IssueTicket.assigned_to', back_populates="assigned_worker")
    resolved_issues = relationship("IssueTicket", foreign_keys='IssueTicket.resolved_by', back_populates="resolving_worker")
    inspection_captures = relationship("InspectionCapture", back_populates="worker")


class UserSession(Base):
    __tablename__ = 'user_sessions'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('app_users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    device_info = Column(Text)
    ip_address = Column(String(45))
    is_revoked = Column(Boolean, default=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())

    user = relationship("AppUser", back_populates="sessions")


class SubscriptionPlan(Base):
    __tablename__ = 'subscription_plans'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    max_spaces = Column(Integer, nullable=False)
    max_workers = Column(Integer, nullable=False)
    ai_analysis_frequency = Column(String(100))
    price_monthly = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())

    properties = relationship("Property", back_populates="subscription_plan")


class Property(Base):
    __tablename__ = 'properties'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey('app_users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    address = Column(Text)
    timezone = Column(String(100))
    verification_status = Column(SQLEnum(VerificationStatus), default=VerificationStatus.PENDING)
    subscription_plan_id = Column(String(36), ForeignKey('subscription_plans.id'))
    created_at = Column(DateTime, default=func.now())

    owner = relationship("AppUser", back_populates="owned_properties")
    subscription_plan = relationship("SubscriptionPlan", back_populates="properties")
    workers = relationship("PropertyWorker", back_populates="property")
    spaces = relationship("Space", back_populates="property")
    master_images = relationship("MasterImage", back_populates="property")
    schedules = relationship("InspectionSchedule", back_populates="property")
    issue_tickets = relationship("IssueTicket", back_populates="property")


class PropertyWorker(Base):
    __tablename__ = 'property_workers'

    user_id = Column(String(36), ForeignKey('app_users.id'), primary_key=True)
    property_id = Column(String(36), ForeignKey('properties.id'), primary_key=True)
    worker_status = Column(SQLEnum(PropertyWorkerStatus), default=PropertyWorkerStatus.PENDING)
    created_at = Column(DateTime, default=func.now())

    user = relationship("AppUser", back_populates="worker_profiles")
    property = relationship("Property", back_populates="workers")


class Space(Base):
    __tablename__ = 'spaces'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    property_id = Column(String(36), ForeignKey('properties.id'), nullable=False)
    name = Column(String(255), nullable=False)
    space_type = Column(SQLEnum(SpaceType), nullable=False)
    floor_level = Column(String(50))
    created_at = Column(DateTime, default=func.now())

    property = relationship("Property", back_populates="spaces")
    views = relationship("SpaceView", back_populates="space")
    master_images = relationship("MasterImage", back_populates="space")


class SpaceView(Base):
    __tablename__ = 'space_views'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    space_id = Column(String(36), ForeignKey('spaces.id'), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())

    space = relationship("Space", back_populates="views")
    master_images = relationship("MasterImage", back_populates="space_view")
    inspection_captures = relationship("InspectionCapture", back_populates="space_view")


class Prompt(Base):
    __tablename__ = 'prompts'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    prompt_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    master_images = relationship("MasterImage", back_populates="prompt")


class MasterImage(Base):
    __tablename__ = 'master_images'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    property_id = Column(String(36), ForeignKey('properties.id'), nullable=False)
    space_id = Column(String(36), ForeignKey('spaces.id'), nullable=False)
    space_view_id = Column(String(36), ForeignKey('space_views.id'), nullable=False)
    master_image_url = Column(Text, nullable=False)
    prompt_id = Column(String(36), ForeignKey('prompts.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    property = relationship("Property", back_populates="master_images")
    space = relationship("Space", back_populates="master_images")
    space_view = relationship("SpaceView", back_populates="master_images")
    prompt = relationship("Prompt", back_populates="master_images")


class InspectionSchedule(Base):
    __tablename__ = 'inspection_schedules'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    property_id = Column(String(36), ForeignKey('properties.id'), nullable=False)
    name = Column(String(255), nullable=False)
    schedule_type = Column(SQLEnum(ScheduleType), nullable=False)
    recurrence_type = Column(SQLEnum(RecurrenceType), nullable=False)
    custom_days = Column(String(255)) 
    start_time_minutes = Column(Integer)
    grace_period_minutes = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    property = relationship("Property", back_populates="schedules")
    captures = relationship("InspectionCapture", back_populates="schedule")


class InspectionCapture(Base):
    __tablename__ = 'inspection_captures'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    worker_id = Column(String(36), ForeignKey('app_users.id'), nullable=False)
    schedule_id = Column(String(36), ForeignKey('inspection_schedules.id'), nullable=False)
    space_view_id = Column(String(36), ForeignKey('space_views.id'), nullable=False)
    capture_image_url = Column(Text, nullable=False)
    capture_status = Column(SQLEnum(CaptureStatus), default=CaptureStatus.PENDING)
    capture_time = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    worker = relationship("AppUser", back_populates="inspection_captures")
    schedule = relationship("InspectionSchedule", back_populates="captures")
    space_view = relationship("SpaceView", back_populates="inspection_captures")
    issues = relationship("IssueTicket", back_populates="capture")


class IssueTicket(Base):
    __tablename__ = 'issue_tickets'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    property_id = Column(String(36), ForeignKey('properties.id'), nullable=False)
    capture_id = Column(String(36), ForeignKey('inspection_captures.id'), nullable=False)
    description = Column(Text, nullable=False)
    issue_status = Column(SQLEnum(IssueStatus), default=IssueStatus.OPEN)
    issue_severity = Column(SQLEnum(IssueSeverity), default=IssueSeverity.MEDIUM)
    assigned_to = Column(String(36), ForeignKey('app_users.id'))
    resolved_by = Column(String(36), ForeignKey('app_users.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    property = relationship("Property", back_populates="issue_tickets")
    capture = relationship("InspectionCapture", back_populates="issues")
    assigned_worker = relationship("AppUser", foreign_keys=[assigned_to], back_populates="assigned_issues")
    resolving_worker = relationship("AppUser", foreign_keys=[resolved_by], back_populates="resolved_issues")