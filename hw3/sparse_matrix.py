from copy import deepcopy

EPS = 10e-12

class SparseMatrix:
    def __init__(self, rows = 0, cols = 0, values = None):
        self.rows = rows
        self.cols = cols
        self.values = values
        if values is None:
            self.values = [[]]

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            print("Matrices have different sizes")
            return None
        else:
            values = deepcopy(self.values)
            for i in range(0, self.rows + 1):
                if len(other.values[i]):
                    for j in range(0, len(other.values[i])):
                        new_value = True
                        for k in range(0, len(values[i])):
                            if values[i][k][1] == other.values[i][j][1]:
                                values[i][k][0] += other.values[i][j][0]
                                new_value = False
                                break
                        if new_value:
                            values[i].insert(0, other.values[i][j])
            return SparseMatrix(self.rows, self.cols, values)

    def __eq__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            print("Matrices have different sizes")
            return False
        else:
            for i in range(0, self.rows + 1):
                if len(self.values[i]) != len(other.values[i]):
                    return False
                self.values[i].sort(key = lambda x:x[1])
                other.values[i].sort(key = lambda x:x[1])
                for j in range(0, len(self.values[i])):
                    if self.values[i][j][1] != other.values[i][j][1] or \
                        abs(self.values[i][j][0] - other.values[i][j][0]) > EPS:
                            print(self.values[i][j][0], other.values[i][j][0])
                            return False
            return True

    def insert(self, value, row, col):
        if row > self.rows:
            for i in range(row - self.rows):
                self.values.append([])
            self.rows = row
        if col > self.cols:
            self.cols = col
        for i in range(0, len(self.values[row])):
            if self.values[row][i][1] == col:
                self.values[row][i] = [self.values[row][i][0] + value, self.values[row][i][1]]
                return
        self.values[row].append([value, col])