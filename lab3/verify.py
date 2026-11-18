import numpy

A = numpy.loadtxt("matrix_a2000.txt", skiprows=1, dtype=numpy.int64)
B = numpy.loadtxt("matrix_b2000.txt", skiprows=1, dtype=numpy.int64)
C_cpp = numpy.loadtxt("matrix_c2000.txt", skiprows=1, dtype=numpy.int64)

C_numpy = A @ B

if numpy.array_equal(C_numpy, C_cpp):
    print("Корректно")
else:
    print("Некорректно")