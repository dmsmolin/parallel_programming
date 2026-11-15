import matplotlib.pyplot
import numpy

matplotlib.pyplot.rcParams['font.family'] = 'DejaVu Sans'

procs = numpy.array([1, 2, 4, 8])
sizes = [200, 400, 800, 1200, 1600, 2000]

times = {
    200:  numpy.array([0.0098, 0.0058, 0.0029, 0.0015]),
    400:  numpy.array([0.0705, 0.0359, 0.0190, 0.0105]),
    800:  numpy.array([0.5578, 0.2811, 0.1429, 0.0763]),
    1200: numpy.array([1.8682, 0.9367, 0.4857, 0.2638]),
    1600: numpy.array([4.1433, 2.2036, 1.1989, 0.6175]),
    2000: numpy.array([8.2434, 4.2410, 2.2391, 1.1939]),
}

colors = ['seagreen', 'yellowgreen', 'gold', 'orange', 'orangered', 'firebrick']

fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 6))

for color, n in zip(colors, sizes):
    ax1.plot(procs, times[n], 'o-', color=color,
             linewidth=2, markersize=8, label=f'N = {n}')

ax1.set_yscale('log')
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax1.set_xlabel('Число процессов (MPI)', fontsize=13)
ax1.set_ylabel('Время, сек (лог. шкала)', fontsize=13)
ax1.set_title('Зависимость времени выполнения от числа процессов', fontsize=14)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)

for color, n in zip(colors, sizes):
    speedup = times[n][0] / times[n]
    ax2.plot(procs, speedup, 'o-', color=color,
             linewidth=2, markersize=8, label=f'N = {n}')

ax2.plot(procs, procs, 'k--', linewidth=2, label='Идеал')

ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax2.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax2.set_xlabel('Число процессов (MPI)', fontsize=13)
ax2.set_ylabel('Ускорение', fontsize=13)
ax2.set_title('Зависимость ускорения от числа процессов для разных размеров матриц', fontsize=14)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig('graph_processes.png', dpi=150)
matplotlib.pyplot.show()