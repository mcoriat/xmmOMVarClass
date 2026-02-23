"""
Plot KDE-smoothed probability distributions from .dat files produced by
makedistrib.py.  Adapted for the OM 3-class system (Star, QSO, Galaxy).
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from matplotlib import rcParams


if __name__ == "__main__":
    rcParams.update({'font.size': 11})

    datadir = 'classif/distrib_KDE_OM/'

    files = glob.glob(datadir + '*dat')

    if len(sys.argv) > 1 and '%s%s.dat' % (datadir, sys.argv[1]) in files:
        files = ['%s%s.dat' % (datadir, sys.argv[1])]
    print(files)

    Class = ['Star', 'QSO', 'Galaxy']
    col = ['C0', 'C1', 'C2']

    os.makedirs('plots/', exist_ok=True)

    for f in files:
        plt.figure(figsize=(7.5, 4.5))
        d = np.loadtxt(f).reshape((-1, 5))
        for i in [2, 3, 4]:
            plt.plot((d[:, 0] + d[:, 1]) / 2, d[:, i] / np.sum(d[:, i]) / (d[1, 0] - d[0, 0]),
                     label=Class[i - 2], color=col[i - 2], alpha=0.8, ls="-")

        plt.legend()
        xlab = f.split('/')[-1].split('.')[0]
        plt.xlabel(xlab)
        plt.subplots_adjust(bottom=0.18, top=0.99, left=0.1, right=0.99)
        plt.savefig('plots/OM_%s.png' % xlab)
        plt.show()
