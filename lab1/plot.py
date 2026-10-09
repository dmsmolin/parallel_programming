import matplotlib.pyplot
import numpy

matplotlib.pyplot.rcParams['font.family'] = 'DejaVu Sans'

N = numpy.array([200, 400, 800, 1600, 2000])
T = numpy.array([0.0045, 0.036, 0.2937, 2.5644, 4.9657])

n_smooth = numpy.linspace(N.min(), N.max(), 300)
t_theory = T[0] * (n_smooth / N[0]) ** 3

matplotlib.pyplot.figure(figsize=(10, 6))

matplotlib.pyplot.plot(N, T, 'o-', color='steelblue', linewidth=2,
                       markersize=10, label='Эксперимент')
matplotlib.pyplot.plot(n_smooth, t_theory, '--', color='orange', linewidth=2,
                       label='Теория O(N³)')

ticks = [200, 400, 800, 1200, 1600, 2000]
matplotlib.pyplot.xticks(ticks, [str(t) for t in ticks])

matplotlib.pyplot.xlabel('Размер матрицы N', fontsize=13)
matplotlib.pyplot.ylabel('Время, сек', fontsize=13)
matplotlib.pyplot.title('Зависимость времени умножения матриц от размера', fontsize=14)
matplotlib.pyplot.grid(True, which='both', linestyle='--', alpha=0.6)
matplotlib.pyplot.legend(fontsize=12)

matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig('graph.png', dpi=150)
matplotlib.pyplot.show()
