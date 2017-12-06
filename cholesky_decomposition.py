import numpy as np



def inverse_finding(A):
    """
    This method is made to find Inverse of symmetric and positive define matrix A, using Cholesky Decomposition method
    The advantage of this approach is that it avoid some unnecessary computations that makes it`s complexity (1/3)N^3
    Other algorithms of finding matrix  inverse are taking +-(5/6)N^3
    :param A: Symmetric, positive define matrix A
    :return: Inverse of this matrix using Cholesky Decomposition method
    """

    help_matrix = find_L(A)

    if help_matrix is False:
        print("Some error with matrix")
        return False

    s = [[0 for i in range(len(help_matrix))] for i in range(len(help_matrix))]
    inverse = [[0 for i in range(len(A))] for i in range(len(A))]
    """
    s - diagonal matrix, with elements which are  reciprocals of diagonal entries of lower triangular matrix
    inverse - Resulting matrix
    """
    x = []

    for i in range(len(s)):
        s[i][i] = 1/help_matrix[i][i]
    """
    Finding elements of matrix s
    """

    help_matrix = find_matrix_transpose(help_matrix)
    """
    Now help_matrix is upper triangular because L^T = U 
    """

    j = len(s)
    for i in range(len(help_matrix)-1, -1, -1):
        x = np.linalg.solve(help_matrix, s[i][:j])
        """
        We are using numpy.linalg.solve(a,b) to solve system of linear equations. 
        By each step of loop, we will get collumn of X-es, which are elements of inverse matrix
        """
        j = j-1
        for j in range(len(x)):
            inverse[j][i] = x[j]
            inverse[i][j] = x[j]
            for k in range(j, -1, -1):
                """
                After finding each element of inverse we must "fix" the coefficients of matrix s
                for further correctnes of computations 
                """
                s[j][k] = s[j][k]-x[j]*help_matrix[k][i]
        help_matrix = cut_matrix(help_matrix)


    for row in inverse:
        for elem in row:
            print(elem, end=' ')
        print()
    return inverse


def find_L(A):
    """
    This method finds L for Cholesky decomposition.
    It takes any matrix and return lower-triangular matrix.
    If matrix doesn't satisfy (n x n) or it isn't symmetric
    method will return False.
    >>> find_L([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    [[2.0, 0, 0], [6.0, 1.0, 0], [-8.0, 5.0, 3.0]]
    """
    for i in range(len(A)):
        if len(A) != len(A[i]):
            return False
    if A != find_matrix_transpose(A):
        return False
    L = [[0 for i in range(len(A))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                L[i][i] = (A[i][i] - find_sum_1(L, i)) ** 0.5
            elif j < i:
                L[i][j] = 1 / L[j][j] * (A[i][j] - find_sum2(L, i, j))
    return L


def find_matrix_transpose(A):
    """
    This method takes any matrix and makes its transponse.
    :param A: any matrix.
    :return: a transponse of current matrix.
    """
    A = [list(i) for i in zip(*A)]
    return A


def find_sum_1(L, i):
    result = 0
    for k in range(i):
        result += L[i][k] ** 2
    return result


def find_sum2(L, i, j):
    result = 0
    for k in range(j):
        result += L[i][k] * L[j][k]
    return result

def cut_matrix(m):
    """
    Method decreases the dimention of matrix by 1
    :param m: Square matrix of size N x N
    :return: Square matrix of size (N-1)x(N-1)
    """
    res = [[0 for i in range(len(m)-1)] for i in range(len(m)-1)]
    for i in range(len(res)):
        for j in range(len(res)):
            res[i][j] = m[i][j]
    return res
