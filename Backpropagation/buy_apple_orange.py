# coding: utf-8
from layer_naive import *


apple = 100  # Price of Apple
apple_num = 2  # Number of Apple
orange = 150  # Price of Orange
orange_num = 3  # Number of orange
tax = 1.1

# layer
mul_apple_layer = MulLayer()  # Apple local
mul_orange_layer = MulLayer()  # Orange local
add_apple_orange_layer = AddLayer()  # Sum of Local
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)

# backward
dprice = 1  # Backpropagation Start
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4) dall_price = 1.1 dtax = 650
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3) each 1.1
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2) dorange = 3.3 dorange_unm = 165
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1) dapple = 2.2 dapple_num = 110

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)
