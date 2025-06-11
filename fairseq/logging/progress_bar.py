class progress_bar:
    def __init__(self, args, iterable, epoch=None, prefix="", default_log_format=None):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def log(self, stats):
        pass

    def print(self, stats):
        pass

    def close(self):
        pass

