import enum


class PredatorLevel(enum.Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

    def get_speed_factor(self) -> float:
        return (self.value + 0.5) * 0.39

    def get_prob_continue_forward_action(self) -> float:
        return 0.8

    def get_prob_continue_rotation_action(self) -> float:
        return 0.1

    def get_prob_follow_agents(self) -> float:
        return (self.value + 0.5) / 3.5

    def get_inactive_steps_after_eating(self) -> int:
        return (3 - self.value) * 10
