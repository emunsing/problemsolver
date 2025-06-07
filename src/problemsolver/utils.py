
class Interval:
    """
    Optuna metadata class for use with parameter annotations using typing.Annotated
    """
    def __init__(self, low, high, step=None, log=False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log