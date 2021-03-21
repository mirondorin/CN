import numpy as np
import sparse_matrix as spm
EPS = 0

matrix = spm.SparseMatrix()
vec_a = []
vec_b = []
vec_c = []
vec_f = []
x_gs = []
n_size = 0
p_size = 0
q_size = 0

def parse_tri_matrix(file_name):
    global x_gs, n_size, p_size, q_size
    with open(file_name, 'r') as f:
        n_size = int(f.readline())
        p_size, q_size = int(f.readline()), int(f.readline())
        EPS = 1 / (10 ** p_size)
        x_gs = [0] * n_size
        f.readline() # Read empty line

        for i in range(0, n_size):
            value = float(f.readline())
            vec_a.append(value)
            if abs(value) < EPS:
                print("Matrix has 0 value in main diagonal")
                exit(0)
            # matrix.insert(value, i, i)

        f.readline() # Read empty line. Start storing (n-p+1) values of c vector
        c_size = n_size - p_size
        row, col = 0, p_size
        for i in range(0, c_size):
            value = float(f.readline())
            vec_c.append(value)
            # matrix.insert(value, row, col)
            row += 1
            col += 1

        f.readline() # Read empty line. Start storing (n-q+1) values of b vector
        b_size = n_size - q_size
        row, col = q_size, 0
        for i in range(0, b_size):
            value = float(f.readline())
            vec_b.append(value)
            # matrix.insert(value, row, col)
            row += 1
            col += 1

def parse_f(file_name):
    with open(file_name, 'r') as f:
        n_size = int(f.readline())
        f.readline() # Read empty line

        for i in range(0, n_size):
            value = float(f.readline())
            vec_f.append(value)

def solve(a_file, f_file):
    parse_tri_matrix(a_file)
    parse_f(f_file)
    while True:
        norm = 0
        for i in range(0, n_size):
            c_sum = 0
            b_sum = 0
        break


if __name__ == "__main__":
    solve("a1.txt", "f1.txt")