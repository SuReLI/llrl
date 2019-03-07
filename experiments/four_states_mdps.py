from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations_with_replacement

"""
Functions representing U_RMAX - U
"""
g = 0.99  # Gamma


def f0(x, y):
	u_rmax = x + g * x + g * g / (1. - g)
	u = abs(x - y) * (1. + g) + (y + g * g * max(y, 1. - y)) / (1. - g)
	return u_rmax - u


def f1(x, y):
	u_rmax = x + g / (1. - g)
	u = abs(x - y) + (y + g * max(y, 1. - y)) / (1. - g)
	return u_rmax - u


def f2(x, y):
	u_rmax = (x - x) + 1 / (1. - g)
	u = (y + max(y, 1. - y)) / (1. - g)
	return u_rmax - u


def f3(x, y):
	u_rmax = (x - x) + 1 / (1. - g)
	u = (y + max(y, 1. - y)) / (1. - g)
	return u_rmax - u


def apply(f, x, y):
	n = len(x)
	z = np.zeros(shape=(n,n))
	for i in range(n):
		for j in range(n):
			z[i, j] = f(x[i, j], y[i, j])
	return z


def plot_upper_bounds_comparison():
	f, axes = plt.subplots(2, 2, sharex='all', sharey='all')

	inf = 1e99
	delta = 0.01
	x, y = np.meshgrid(np.arange(0.0, 1.0, delta), np.arange(0.0, 1.0, delta))
	lv = [-inf, -.0001, .0001, inf]
	cl = ['orange', 'salmon', 'teal']

	axes[0, 0].contourf(x, y, apply(f0, x, y), levels=lv, colors=cl)
	axes[0, 1].contourf(x, y, apply(f1, x, y), levels=lv, colors=cl)
	axes[1, 1].contourf(x, y, apply(f2, x, y), levels=lv, colors=cl)
	axes[1, 0].contourf(x, y, apply(f3, x, y), levels=lv, colors=cl)

	axes[0, 0].set_title(r'$(s_0, a)$')
	axes[0, 1].set_title(r'$(s_1, a)$')
	axes[1, 1].set_title(r'$(s_2, a)$')
	axes[1, 0].set_title(r'$(s_3, a)$')

	plt.rc('text', usetex=True)
	for i in list(combinations_with_replacement([0, 1, 0], 2)):
		axes[i].set_xlabel(r'R')
		axes[i].set_ylabel(r'$\bar{R}$')

	legend_elements = [
		Patch(facecolor=cl[0], label=r'$U_{RMAX} < U$'),
		Patch(facecolor=cl[1], label=r'$U = U_{RMAX}$'),
		Patch(facecolor=cl[2], label=r'$U < U_{RMAX}$')
	]
	axes[1, 0].legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0., -.75))
	plt.subplots_adjust(bottom=0.25)
	plt.show()


if __name__ == "__main__":
	plot_upper_bounds_comparison()

