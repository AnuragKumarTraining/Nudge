from enum import Enum


class PropertyType(str, Enum):
    HOTEL = "HOTEL"
    CAFE = "CAFE"
    PG = "PG"
    CO_LIVING = "CO_LIVING"
    RESORT = "RESORT"
    SHORT_TERM_RENTAL = "SHORT_TERM_RENTAL"