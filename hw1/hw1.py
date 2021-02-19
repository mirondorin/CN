import random
import math  

def ex1():
    for iterations in range (1000, 0, -1):
        u = pow(10, -iterations)
        if 1.0 + u != 1.0:
            # print(u, iterations)
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
    
def ex2_2():
    for it in range(0, 1000):
        a = random.random()
        b = random.random()
        c = random.random()
        if (a*b)*c != a*(b*c):
            print("multiplication is non-associative", a, b, c)
            break

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

def ex3():
    for i in range (0, 10000):
        x = random.uniform(-math.pi/2, math.pi/2)
        print(lentz(x, 1e-15), mac_laurin(x), math.tan(x))

print(ex1())
ex2_1()
ex2_2()
ex3()