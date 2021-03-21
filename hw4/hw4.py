import numpy as np
import math
import sparse_matrix as spm
EPS = 10e-12

matrix = spm.SparseMatrix()
vec_a = []
vec_b = []
vec_c = []
vec_f = []
n_size = 0
p_size = 0
q_size = 0

def calculate_norm(x, y):
    if len(x) != len(y):
        return None
    return np.linalg.norm(np.asarray(x) - np.asarray(y))

def parse_tri_matrix(file_name):
    global matrix, vec_f, n_size, p_size, q_size
    matrix = spm.SparseMatrix()
    vec_f = []
    with open(file_name, 'r') as f:
        n_size = int(f.readline())
        p_size, q_size = int(f.readline()), int(f.readline())
        # EPS = 1 / (10 ** p_size)
        f.readline() # Read empty line

        for i in range(0, n_size):
            value = float(f.readline())
            vec_a.append(value)
            if abs(value) < EPS:
                print("Matrix has 0 value in main diagonal")
                exit(0)
            matrix.insert(value, i, i)

        f.readline() # Read empty line. Start storing (n-p+1) values of c vector
        c_size = n_size - p_size
        row, col = 0, p_size
        for i in range(0, c_size):
            value = float(f.readline())
            vec_c.append(value)
            matrix.insert(value, row, col)
            row += 1
            col += 1

        f.readline() # Read empty line. Start storing (n-q+1) values of b vector
        b_size = n_size - q_size
        row, col = q_size, 0
        for i in range(0, b_size):
            value = float(f.readline())
            vec_b.append(value)
            matrix.insert(value, row, col)
            row += 1
            col += 1

def parse_f(file_name):
    with open(file_name, 'r') as f:
        n_size = int(f.readline())
        f.readline() # Read empty line

        for i in range(0, n_size):
            value = float(f.readline())
            vec_f.append(value)

# def solve_old(a_file, f_file):
#     parse_tri_matrix(a_file)
#     parse_f(f_file)
#     k = 0
#     k_max = 10000
#     while True:
#         norm = 0
#         for i in range(0, n_size):
#             c_sum = 0
#             b_sum = 0
#             for j in range(0, i):
#                 if i >= p_size and i - j == p_size: #diag jos 
#                     c_sum += vec_c[j - p_size] * x_gs[j]
#                     # print("i, j", (i, j, vec_c[j]))
#             for j in range(i + 1, n_size):
#                 if j >= q_size and j - i == q_size:
#                     b_sum += vec_b[j - q_size] * x_gs[j]
#                     # print("i, j", (i, j, vec_b[j-q_size]))
#             prev_x = x_gs[i]
#             x_gs[i] = (vec_f[i] - c_sum - b_sum) / vec_a[i]
#             norm += (prev_x - x_gs[i]) ** 2
#         if norm == math.inf:
#             print("Solution diverges")
#             break
#         if norm < EPS or k >= k_max:
#             break
#         k += 1

#     sol_norm = calculate_norm(x_gs * matrix, vec_f)
#     print("Iterations: ", k)
#     print("Final norm: ", sol_norm)

def solve(a_file, f_file):
    parse_tri_matrix(a_file)
    parse_f(f_file)
    x_gs = [0] * n_size
    k = 0
    k_max = 100
    while True:
        norm = 0
        for i in range(0, n_size):
            c_sum = 0
            b_sum = 0
            m_diag = 0
            for val, col in matrix.values[i]:
                if col < i:
                    c_sum += val * x_gs[col]
                elif col > i:
                    b_sum += val * x_gs[col]
                elif col == i:
                    m_diag = val
            prev_x = x_gs[i]
            x_gs[i] = (vec_f[i] - c_sum - b_sum) / m_diag
            norm += (prev_x - x_gs[i]) ** 2
        if norm == math.inf:
            print("Solution diverges")
            break

        if norm < EPS or k > k_max:
            break
        k += 1

    sol_norm = calculate_norm(x_gs * matrix, vec_f)
    print("Solution for: ", a_file)
    print("Iterations: ", k - 1)
    print("Final norm: ", sol_norm)
    print("")

# TODO make exception for results too large.
if __name__ == "__main__":
    solve("a1.txt", "f1.txt")
    solve("a2.txt", "f2.txt")
    solve("a3.txt", "f3.txt")
    solve("a4.txt", "f4.txt")
    solve("a5.txt", "f5.txt")