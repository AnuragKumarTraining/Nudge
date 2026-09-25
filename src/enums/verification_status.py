from enum import Enum


class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    ACTION_REQUIRED = "ACTION_REQUIRED"