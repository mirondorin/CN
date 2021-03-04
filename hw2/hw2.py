import numpy as np
import random
import copy
from sklearn import datasets
import copy

def eps():
    for iterations in range (1000, 0, -1):
        u = pow(10, -iterations)
        if 1.0 + u != 1.0:
            return u

EPS = eps()

def calc_transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def get_identity(n):
    m=[[0 for x in range(n)] for y in range(n)]
    for i in range(0,n):
        m[i][i] = 1
    return m

def get_matrix_minor(m,i,j):
    #return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]
    return np.delete(np.delete(m,i,axis=0), j, axis=1)

def calc_determinant(m):
    #base case for 2x2 matrix
    if m.shape[0] == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(m.shape[0]):
        determinant += ((-1)**c)*m[0][c]*calc_determinant(get_matrix_minor(m,0,c))
    return determinant

def get_matrix_inverse(m):
    determinant = calc_determinant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = get_matrix_minor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * calc_determinant(minor))
        cofactors.append(cofactorRow)
    cofactors = calc_transpose(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors


def generate_random_spd(size):
    system = []
    for i in range(0, size):
        current_line = []
        for j in range(0, size):
            current_line.append(random.random())
        system.append(current_line)

    result = []
    for i in range(0, size):
        result.append(random.random())
    system = np.array(system)
    upper_triangle = np.triu(system, 0)
    system = upper_triangle + upper_triangle.T
    return system, result

def cholesky_factorization(system, n, col, lower):
    prev_sum = 0

    for row in range (0, system.shape[0]):
        if col == row:
            for k in range(0, col):
                prev_sum += lower[col][k] ** 2
            lower[row][col] = np.sqrt(system[row][col] - prev_sum)
        elif col < row:
            prev_sum = 0
            for k in range(0, col):
                prev_sum += lower[row][k] * lower[col][k]
            if abs(lower[col][col]) > EPS:
                lower[row][col] = (system[row][col] - prev_sum) / lower[col][col]
            else:
                lower[row][col] = 10e-12

    col += 1
    if col == system.shape[1]: 
        return lower
    return cholesky_factorization(system, n, col, lower)

def x_chol(L, L_t, b_vec):
    n = L.shape[0]
    y_vec = [0 for i in range(0, n)]

    for i in range (0, n):
        value = b_vec[i]
        for j in range (0, i + 1):
            if i == j:
                if abs(L[i][j]) > EPS:
                    y_vec[j] = value / L[i][j]
                else:
                    y_vec[j] = 10e-12
            else:
                value -= L[i][j] * y_vec[j]
    
    x_vec = [0 for i in range(0, n)]
    for i in range (n-1, -1, -1):
        value = y_vec[i]
        for j in range (n-1, i - 1, -1):
            value -= L[j][i] * x_vec[j]
        if abs(L_t[i][i]) > EPS:
            x_vec[j] = value / L[i][i]
        else:
            x_vec[j] = 10e-12
    return x_vec

def verify_sol(matrix, x_chol, result):
    sol = np.matmul(matrix, np.array(x_chol))
    sol = np.subtract(sol, np.array(result))
    norm = 0
    for i in range (0, sol.shape[0]):
        norm += sol[i] ** 2
    return np.sqrt(norm)

def read_matrix_keyboard():
    pass

if __name__ == "__main__":
    matrix, vec = generate_random_spd(8)
    matrix = datasets.make_spd_matrix(8)
    lower = np.zeros((matrix.shape[0], matrix.shape[1]))
    L = cholesky_factorization(matrix, matrix.shape[1], 0, lower)
    L_t = L.T
    if calc_determinant(L) * calc_determinant(L_t) == 0:
        print("Determinant is 0")
        exit(0)
    x_sol = x_chol(L, L_t, vec)
    norm = verify_sol(matrix, x_sol, vec)
    np_norm = np.linalg.solve(matrix, vec)
    #print(norm)
    #print(np.linalg.norm(np_norm))
    inv_custom = np.array(get_matrix_inverse(L))
    inv_np = np.linalg.inv(L)
    #print(inv_custom)
    #print(inv_np)
    print(np.linalg.norm(inv_custom - inv_np))