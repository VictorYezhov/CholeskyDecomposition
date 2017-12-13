import random


def matrix_multiplication(matrix1, matrix2):
    result = [[0 for i in range(len(matrix1))] for i in range(len(matrix2[0]))]
    for i in range(len(result)):
        for j in range(len(result[i])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k]*matrix2[k][j]
    return result


def generate(size):
    """
    Matrix generator
    :param size: int size of matrix, which you want to get
    :return: simmetric, positive-define matrix
    """
    print("Generation")
    matrix =[[0 for i in range(size*random.randint(size,size*size))] for i in range(size)]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = random.randint(1, 9)
    matrix_t = find_matrix_transpose(matrix)
    result = matrix_multiplication(matrix, matrix_t)
    print("End of generation")
    return result


def find_matrix_transpose(A):
    """
    This method takes any matrix and makes its transponse.
    :param A: any matrix.
    :return: a transponse of current matrix.
    """
    A = [list(i) for i in zip(*A)]
    return A




