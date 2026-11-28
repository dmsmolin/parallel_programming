import os
import subprocess
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_matrix import generate_and_save_matrices
from verify import verify_results

SIZES = [200, 400, 800, 1200, 1600, 2000]
BLOCK_SIZES = [8, 16, 32]
CU_SOURCE = "src/main.cu"
CU_BINARY = "main_cuda.exe"

def parse_info_txt(info_path="output/info.txt"):
    time_sec, gflops = 0.0, 0.0
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Execution time:" in line:
                    time_sec = float(line.split(": ")[1].split()[0])
                elif "Performance:" in line:
                    gflops = float(line.split(": ")[1].split()[0])
    return time_sec, gflops

def compile_cuda():
    print("Поиск окружения MSVC для nvcc...")
    vcvars_search = "C:/Program Files/Microsoft Visual Studio/**/vcvars64.bat"
    found_vcvars = glob.glob(vcvars_search, recursive=True)
    if not found_vcvars:
        found_vcvars = glob.glob("C:/Program Files/Microsoft Visual Studio/**/vcvarsall.bat", recursive=True)

    if found_vcvars:
        vc_bat = os.path.abspath(found_vcvars[0])
        compile_cmd = f'"{vc_bat}" amd64 && nvcc -O3 "{CU_SOURCE}" -o "{CU_BINARY}"'
    else:
        compile_cmd = f'nvcc -O3 "{CU_SOURCE}" -o "{CU_BINARY}"'

    print("Компиляция CUDA кода...")
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Ошибка компиляции nvcc:")
        print(result.stderr)
        return False

    for ext in ["*.exp", "*.lib"]:
        for f in glob.glob(ext):
            os.remove(f)
    print(f"Компиляция успешно завершена! Создан {CU_BINARY}")
    return True

def main():
    if not compile_cuda():
        return

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    results = []
    print(f"\n{'Размер':<8} | {'Блок':<8} | {'Время (сек)':<12} | {'GFLOPS':<10} | {'Макс. Ошибка':<15} | {'Статус':<8}")
    print("-" * 78)

    binary_path = os.path.abspath(CU_BINARY)

    for size in SIZES:
        generate_and_save_matrices(size, base_path="input")
        for b_size in BLOCK_SIZES:
            subprocess.run([binary_path, str(b_size)], cwd=".", check=True, capture_output=True)
            is_valid, max_error = verify_results(input_dir="input", output_dir="output")
            time_sec, gflops = parse_info_txt()
            status = "PASS" if is_valid else "FAIL"

            print(f"{size:<8} | {f'{b_size}x{b_size}':<8} | {time_sec:<12.6f} | {gflops:<10.2f} | {max_error:<15.2e} | {status:<8}")

            results.append({
                "Size": size,
                "Block": b_size,
                "Time": time_sec,
                "GFLOPS": gflops,
                "Max_Error": max_error,
                "Verified": status
            })

    with open("output/experiments_cuda.txt", "w", encoding="utf-8") as f:
        f.write("Размер | Блок | Время (сек) | GFLOPS | Макс. Ошибка | Верификация\n")
        f.write("-" * 78 + "\n")
        for r in results:
            f.write(f"{r['Size']} | {r['Block']}x{r['Block']} | {r['Time']:.6f} | {r['GFLOPS']:.2f} | {r['Max_Error']:.2e} | {r['Verified']}\n")

    print("\nВсе эксперименты CUDA успешно завершены!")

if __name__ == "__main__":
    main()