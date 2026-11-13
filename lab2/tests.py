import subprocess
import numpy

EXE = "LAB_2.exe"
SIZES = [200, 400, 800, 1200, 1600, 2000]

def generate_matrix(path, n, low=0, high=9):
    matrix = numpy.random.randint(low, high + 1, size=(n, n))
    numpy.savetxt(path, matrix, fmt="%d", header=str(n), comments="")

def run_cpp():
    result = subprocess.run(
        [EXE],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("  C++ упал с кодом", result.returncode)
        if result.stderr:
            print("  stderr:", result.stderr.strip())
        return None
    return result.stdout

def verify():
    A = numpy.loadtxt("matrix_a.txt", skiprows=1, dtype=numpy.int64)
    B = numpy.loadtxt("matrix_b.txt", skiprows=1, dtype=numpy.int64)
    C_cpp = numpy.loadtxt("matrix_c.txt", skiprows=1, dtype=numpy.int64)
    C_numpy = A @ B
    return numpy.array_equal(C_numpy, C_cpp)

def main():
    all_ok = True

    for n in SIZES:
        print(f"\n========= N = {n} =========")

        generate_matrix("matrix_a.txt", n)
        generate_matrix("matrix_b.txt", n)

        output = run_cpp()
        if output is None:
            all_ok = False
            continue

        ok = verify()
        print("      Верификация:", "OK" if ok else "FAIL")
        if not ok:
            all_ok = False

    print("\n" + "=" * 28)
    print("ИТОГ:", "все проверки пройдены" if all_ok else "есть ошибки")

if __name__ == "__main__":
    main()