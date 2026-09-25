from enum import Enum


class RecurrenceType(str, Enum):
    DAILY = "DAILY"
    CUSTOM_DAYS = "CUSTOM_DAYS"