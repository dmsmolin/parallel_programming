import matplotlib.pyplot
import numpy

matplotlib.pyplot.rcParams['font.family'] = 'DejaVu Sans'

threads = numpy.array([1, 2, 4, 8])
sizes = [200, 400, 800, 1200, 1600, 2000]

times = {
    200:  numpy.array([0.0058, 0.0029, 0.0017, 0.0014]),
    400:  numpy.array([0.0447, 0.0230, 0.0117, 0.0110]),
    800:  numpy.array([0.3577, 0.1755, 0.0903, 0.0868]),
    1200: numpy.array([1.2332, 0.6060, 0.3055, 0.2584]),
    1600: numpy.array([2.9249, 1.4914, 0.7541, 0.6172]),
    2000: numpy.array([5.9836, 2.9039, 1.4793, 1.1355]),
}

colors = ['seagreen', 'yellowgreen', 'gold', 'orange', 'orangered', 'firebrick']

fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 6))

for color, n in zip(colors, sizes):
    ax1.plot(threads, times[n], 'o-', color=color,
             linewidth=2, markersize=8, label=f'N = {n}')

ax1.set_yscale('log')
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax1.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax1.set_xlabel('Число потоков', fontsize=13)
ax1.set_ylabel('Время, сек (лог. шкала)', fontsize=13)
ax1.set_title('Зависимость времени выполнения от числа потоков', fontsize=14)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)

for color, n in zip(colors, sizes):
    speedup = times[n][0] / times[n]
    ax2.plot(threads, speedup, 'o-', color=color,
             linewidth=2, markersize=8, label=f'N = {n}')

ax2.plot(threads, threads, 'k--', linewidth=2, label='Идеал')

ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax2.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
ax2.set_xlabel('Число потоков', fontsize=13)
ax2.set_ylabel('Ускорение', fontsize=13)
ax2.set_title('Зависимость ускорения от числа потоков для разных размеров матриц', fontsize=14)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig('graph_threads.png', dpi=150)
matplotlib.pyplot.show()