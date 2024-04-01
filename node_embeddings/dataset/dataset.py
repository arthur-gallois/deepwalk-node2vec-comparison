class Dataset:
    def __init__(self):
        pass

    def load(self):
        raise NotImplementedError

    def get_evaluator(self):
        raise NotImplementedError