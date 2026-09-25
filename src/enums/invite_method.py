from enum import Enum


class InviteMethod(str, Enum):
    LINK = "LINK"
    CODE = "CODE"