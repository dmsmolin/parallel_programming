import os
import matplotlib.pyplot as plt
from collections import defaultdict

txt_path = "output/experiments_cuda.txt"

if not os.path.exists(txt_path):
    print(f"Ошибка: Файл {txt_path} не найден! Сначала запустите experiments.py.")
    exit(1)

data = defaultdict(dict)
sizes_set = set()
blocks_set = set()

with open(txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines[2:]:
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) >= 4:
        size = int(parts[0])
        block = int(parts[1].split("x")[0])
        time_sec = float(parts[2])
        gflops = float(parts[3])
        data[size][block] = (time_sec, gflops)
        sizes_set.add(size)
        blocks_set.add(block)

sizes = sorted(list(sizes_set))
blocks = sorted(list(blocks_set))

# 1. График: Время от размера матрицы
plt.figure(figsize=(10, 6))
for b in blocks:
    times = [data[s][b][0] for s in sizes]
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=6, label=f'Block {b}x{b}')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время, с', fontsize=12)
plt.title('CUDA: время умножения от размера матрицы', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('output/time_vs_size.png', dpi=150)
plt.close()

# 2. График: Производительность (GFLOPS) от размера блока
plt.figure(figsize=(10, 6))
for s in sizes:
    gflops_list = [data[s][b][1] for b in blocks]
    plt.plot(blocks, gflops_list, 's-', linewidth=2, markersize=6, label=f'N={s}')

plt.xlabel('Размер блока (потоков на сторону)', fontsize=12)
plt.ylabel('GFLOPS', fontsize=12)
plt.title('CUDA: производительность при разных конфигурациях блоков', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('output/gflops_vs_block.png', dpi=150)
plt.close()

print("Графики сохранены в output/time_vs_size.png и output/gflops_vs_block.png\n")

# Вывод готовой Markdown-таблицы для отчета
print("=== Готовая таблица для отчета ===")
print("| N | 8x8, GFLOPS | 16x16, GFLOPS | 32x32, GFLOPS |")
print("| :--- | :--- | :--- | :--- |")
best_gflops = 0
best_cfg = ""
for s in sizes:
    row = f"| {s} |"
    for b in blocks:
        gf = data[s][b][1]
        row += f" {gf:.2f} |"
        if gf > best_gflops:
            best_gflops = gf
            best_cfg = f"N={s}, блок {b}x{b} — {gf:.2f} GFLOPS"
    print(row)
print(f"\nЛучший результат: {best_cfg}.")