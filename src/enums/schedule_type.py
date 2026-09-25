from enum import Enum


class ScheduleType(str, Enum):
    RECURRING = "RECURRING"
    EVENT_DRIVEN = "EVENT_DRIVEN"