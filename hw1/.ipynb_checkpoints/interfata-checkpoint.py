import random
import math

def ex1():
    for iterations in range (1000, 0, -1):
        u = pow(10, -iterations)
        if 1.0 + u != 1.0:
            print("1 + u != 1 failed for u=", u, "after", iterations, "iterations")
            return u

def ex2_1():
    u = ex1()
    a = 1.0
    b = u / 10
    c = u / 10
    if (a+b)+c != a+(b+c):
        print("addition is non-associative")
    else:
        print("addition is associative")
    
    return (a,b,c)
    
def ex2_2():
    for it in range(0, 1000):
        a = random.random()
        b = random.random()
        c = random.random()
        if (a*b)*c != a*(b*c):
            print("multiplication is non-associative", a, b, c)
            break
    return (a,b,c)


def lentz(x, eps):
    x %= math.pi
    b = 0
    f_prev = b
    a = x
    mic = pow(10, -12)
    if f_prev == 0:
        f_prev = mic
    C_prev = f_prev
    D_prev = 0
    b = 1
    while True:
        D = b + a * D_prev
        if D == 0:
            D = mic
        C = b + a / C_prev
        if C == 0:
            C = mic
        D = 1 / D
        delta = C * D
        f = delta * f_prev
        a = -(pow(x, 2))
        b += 2
        C_prev = C
        D_prev = D
        f_prev = f
        if(abs(delta - 1) < eps):
            return f

def mac_laurin(x):
    x %= math.pi
    inverse = False
    if x > math.pi / 4:
        x = math.pi / 2 - x
        inverse = True
    res = x + 1 / 3 * pow(x, 3) + 2 / 15 * pow(x, 5) + 17 / 315 * pow(x, 7) + 62 / 2835 * pow(x, 9)
    if inverse == True:
        return 1 / res
    else:
        return res

    
ex3_resultlist = []
def ex3():
    for i in range (0, 10000):
        x = random.uniform(-math.pi/2, math.pi/2)
        #print(lentz(x, 1e-15), mac_laurin(x), math.tan(x))
        ex3_resultlist.append((lentz(x, 1e-15), mac_laurin(x), math.tan(x)))
        
v1 = ex1()
v2 = ex2_1()
v3 = ex2_2()
ex3()

import ipywidgets as widgets
from IPython import display
import pandas as pd
import numpy as np


# sample data
df1 = pd.DataFrame({'a':[v2[0],v3[0]], 'b':[v2[1],v3[1]], 'c':[v2[2],v3[2]]})
df2 = pd.DataFrame(ex3_resultlist, columns=['lentz','mac_laurin','tan'])
# create output widgets
widget1 = widgets.Output()
widget2 = widgets.Output()
# render in output widgets
with widget1:
    display.display(df1)
with widget2:
    display.display(df2)

# create HBox
hbox = widgets.HBox([widget1,widget2])
# render hbox
hbox