import enum

class Role(enum.Enum):
    OWNER = "OWNER"
    WORKER = "WORKER"

class CaptureStatus(enum.Enum):
    PENDING = "PENDING"
    OK = "OK"
    REJECTED = "REJECTED"

class VerificationStatus(enum.Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    ACTION_REQUIRED = "ACTION_REQUIRED"

class ScheduleType(enum.Enum):
    RECURRING = "RECURRING"
    EVENT_DRIVEN = "EVENT_DRIVEN"

class RecurrenceType(enum.Enum):
    DAILY = "DAILY"
    CUSTOM_DAYS = "CUSTOM_DAYS"

class SpaceType(enum.Enum):
    GUEST_ROOM = "GUEST_ROOM"
    BEDROOM = "BEDROOM"
    KITCHEN = "KITCHEN"
    BAR_COUNTER = "BAR_COUNTER"
    DINING_AREA = "DINING_AREA"
    OUTDOOR = "OUTDOOR"
    RESTROOM = "RESTROOM"
    RECEPTION = "RECEPTION"
    STORAGE = "STORAGE"
    OTHER = "OTHER"

class PropertyWorkerStatus(enum.Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"

class IssueStatus(enum.Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"

class IssueSeverity(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"