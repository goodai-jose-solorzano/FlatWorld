import enum


class AgentColoring(enum.Enum):
    PRIMARY = 0
    SECONDARY = 1

    def get_front(self):
        if self == AgentColoring.PRIMARY:
            return 0xC0, 0xF8, 0xD2, 0xFF,
        else:
            return 0xDF, 0xC1, 0xDF, 0xFF,

    def get_back(self):
        if self == AgentColoring.PRIMARY:
            return 0xA0, 0xDD, 0xB2, 0xFF,
        else:
            return 0xC8, 0xA5, 0xC8, 0xFF,

    def opposite(self):
        return AgentColoring.SECONDARY if self == AgentColoring.PRIMARY else AgentColoring.PRIMARY
