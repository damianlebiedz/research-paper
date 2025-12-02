from dataclasses import dataclass


@dataclass
class PositionState:
    z_score: float = None
    spread: float = None
    beta: float = None
    mean: float = None
    std: float = None

    position: float = 0
    prev_position: float = 0
    q_x: float = 0
    q_y: float = 0
    w_x: float = None
    w_y: float = None
    entry_val: float = 0
    stop_loss_threshold: float = None

    def update_hedge_if_none(self, z_score, spread, beta, mean, std):
        if self.z_score is None: self.z_score = z_score
        if self.spread is None: self.spread = spread
        if self.beta is None: self.beta = beta
        if self.mean is None: self.mean = mean
        if self.std is None: self.std = std

    def update_position(self, position, prev_position, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold):
        self.position = position
        self.prev_position = prev_position
        self.q_x = q_x
        self.q_y = q_y
        self.w_x = w_x
        self.w_y = w_y
        self.entry_val = entry_val
        self.stop_loss_threshold = stop_loss_threshold

    def clear_hedge(self):
        self.z_score = None
        self.spread = None
        self.beta = None
        self.mean = None
        self.std = None

    def clear_position(self):
        self.position = 0
        self.q_x = 0
        self.q_y = 0
        self.w_x = None
        self.w_y = None
        self.entry_val = 0
        self.stop_loss_threshold = None


@dataclass
class StrategyParams:
    entry_threshold: float
    exit_threshold: float
    stop_loss: float
    fee_rate: float
    initial_cash: float
