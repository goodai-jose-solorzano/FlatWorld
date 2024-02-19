import enum


class ActuatorType(enum.Enum):
    DEFAULT = 0
    BOOSTERS_2 = 1
    BOOSTERS_4 = 2
    DUAL = 3

    def get_num_boosters(self):
        if self == ActuatorType.DEFAULT or self == ActuatorType.DUAL:
            return 0
        elif self == ActuatorType.BOOSTERS_2:
            return 2
        elif self == ActuatorType.BOOSTERS_4:
            return 4
        else:
            raise Exception(f'Unknown actuator type: {self}')
