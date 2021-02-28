import numpy as np
import random
import copy

def eps():
    for iterations in range (1000, 0, -1):
        u = pow(10, -iterations)
        if 1.0 + u != 1.0:
            return u

def generate_random_spd(size):
    system = []
    for i in range(0, size):
        current_line = []
        for j in range(0, size):
            current_line.append(random.random() / 10e9)
        system.append(current_line)

    result = []
    for i in range(0, size):
        result.append(random.random() / 10e9)
    system = np.array(system)
    upper_triangle = np.triu(system, -1)
    system = upper_triangle + upper_triangle.T
    return system, result

def cholesky_factorization(system):
    L = copy.deepcopy(system)
    L_t = copy.deepcopy(system)
    for i in range (0, system.shape[0]):
        for j in range (0, system.shape[1]):
            if i < j:
                L[i][j] = 0
            elif i > j:
                L_t[i][j] = 0
    return (L, L_t)

def x_chol(L, L_t, result):
    y = result[0] / L[0][0]
    y_vec = [y]
    for i in range (1, L.shape[0]):
        y_prev_sum = 0
        for j in range (0, i):
            y_prev_sum += y_vec[j] * L[i][j]
        y = (result[i] - y_prev_sum) / L[i][i]
        y_vec.append(y)
    
    x = y_vec[-1] / L_t[-1][-1]
    x_vec = [x]
    for i in range (L_t.shape[0]-2, -1, -1):
        idx = -1
        x_prev_sum = 0
        for j in range (L_t.shape[1]-1, i, -1):
            x_prev_sum += x_vec[idx] * L_t[i][j]
            idx -= 1
        x = (y_vec[i] - x_prev_sum) / L_t[i][j]
        x_vec.insert(0, x)
    return y_vec

def verify_sol(matrix, x_chol, result):
    sol = np.matmul(matrix, np.array(x_chol))
    sol = np.subtract(sol, np.array(result))
    norm = 0
    for i in range (0, sol.shape[0]):
        norm += sol[i]**2
    return np.sqrt(norm)

if __name__ == "__main__":
    matrix, vec = generate_random_spd(3)
    L, L_t = cholesky_factorization(matrix)
    L_det = np.linalg.det(L)
    L_t_det = np.linalg.det(L_t)
    if L_det * L_t_det == 0:
        print("Determinant is 0")
        exit(0)
    x_chol = x_chol(L, L_t, vec)
    norm = verify_sol(matrix, x_chol, vec)
    print(norm)
