# 枚举类
from enum import Enum

class Incident(Enum):
    OVER_SPEED = "超速"
    BACK_UP = "逆行/倒车"
    TOO_CLOSE_PRE = "与前车车距过近"
    EMERGENCY_STOP = "紧急停车"
    JAM = "拥堵"