from enum import Enum


class CaptureStatus(str, Enum):
    PENDING = "PENDING"
    OK = "OK"
    REJECTED = "REJECTED"