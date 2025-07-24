from pettingzoo.mpe import simple_spread_v3


class MPEWarpper:
    def __init__(self, ori_env) -> None:
        self.env = ori_env
