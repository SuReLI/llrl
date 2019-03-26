import llrl.utils.distribution as dist
import numpy as np


class Foo(object):
    def __init__(self):
        self.name = "Foo"

    def method(self):
        print(self.name)


class Bar(Foo):
    def __init__(self):
        Foo.__init__(self)
        self.name = "Bar"

    def method(self):
        print("My bar-name is", self.name)


def test():
    f = Foo()
    b = Bar()

    f.method()
    b.method()


if __name__ == "__main__":
    test()
