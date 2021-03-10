# coding: utf-8

#   x, y -> Multiply -> xy
#   dout: dL/dz, if xy is z
#   dz/dx = y * dout = dL/dx 
#   dz/dy = x * dout = dL/dy
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy

#   x, y -> Add -> x+y
#   dout: dL/dz, if x+y is z
#   dz/dx = 1 * dout 
#   dz/dy = 1 * dout
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy