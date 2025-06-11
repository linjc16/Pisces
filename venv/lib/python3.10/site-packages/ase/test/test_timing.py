import time

from ase.utils.timing import Timer, timer


class A:
    def __init__(self):
        self.timer = Timer()

    @timer('run')
    def run(self):
        for i in self.yielding():
            print(i)

    @timer('yield')
    def yielding(self):
        for i in range(5):
            time.sleep(0.001)
            yield i


def test_timer():
    a = A()
    a.run()
    a.timer.write()
    t = a.timer.timers
    ty = t[('run', 'yield')]
    assert ty > 0.005, t
