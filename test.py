import math

class Test(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = None

    def sum(self):
        self.c = self.a + self.b
        return self.c

if __name__ == "__main__":
    from test import Test
    test = Test(2, 3)
    for _ in range(2):
        sum = test.sum()
        print(sum)
