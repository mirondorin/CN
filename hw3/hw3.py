import sparse_matrix as spm

a_matrix = spm.SparseMatrix()
b_matrix = spm.SparseMatrix()
aplusb_matrix = spm.SparseMatrix()
aplusb_file_matrix = spm.SparseMatrix()

def parse_normal_matrix(file_name, matrix):
    with open(file_name, 'r') as f:
        n_size = int(f.readline())
        f.readline() # Read empty line

        while True:
            line = f.readline().strip()

            if not line:
                break
            else:
                data = [float(x) for x in line.split(', ')]
                matrix.insert(float(data[0]), int(data[1]), int(data[2]))

def parse_tri_matrix(file_name, matrix):
    with open(file_name, 'r') as f:
        n_size = int(f.readline())
        p_size, q_size = int(f.readline()), int(f.readline())
        f.readline() # Read empty line

        for i in range(0, n_size):
            value = float(f.readline())
            matrix.insert(value, i, i)

        f.readline() # Read empty line. Start storing (n-p+1) values of b vector
        b_size = n_size - p_size
        row, col = 0, p_size
        for i in range(0, b_size):
            value = float(f.readline())
            matrix.insert(value, row, col)
            row += 1
            col += 1

        f.readline() # Read empty line. Start storing (n-q+1) values of c vector
        c_size = n_size - q_size
        row, col = q_size, 0
        for i in range(0, c_size):
            value = float(f.readline())
            matrix.insert(value, row, col)
            row += 1
            col += 1

parse_normal_matrix("a.txt", a_matrix)
parse_tri_matrix("b.txt", b_matrix)
parse_normal_matrix("aplusb.txt", aplusb_file_matrix)
aplusb_matrix = a_matrix + b_matrix
print(aplusb_matrix == aplusb_file_matrix)