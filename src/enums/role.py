from enum import Enum


class Role(str, Enum):
    OWNER = "OWNER"
    WORKER = "WORKER"