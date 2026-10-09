import numpy

def generate_matrix(path, n, low=0, high=9):
    matrix = numpy.random.randint(low, high + 1, size=(n, n))
    numpy.savetxt(path, matrix, fmt="%d", header=str(n), comments="")

n = int(input("Введите размер матриц (N): "))
generate_matrix("matrix_a.txt", n)
generate_matrix("matrix_b.txt", n)
print(f"Готово! Размер {n}x{n}")