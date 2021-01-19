import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp
import gs2_plotting as gplot


def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    phi2z = np.squeeze(myout['phi2_by_mode'][:,0,:])

    # Reorder kx to be increasing, leaving out kx=0
    phi2z = np.concatenate((phi2z[:,mygrids.nxmid:], phi2z[:,1:mygrids.nxmid]), axis=1)

    # kx grid excluding zero
    kxnozero = np.concatenate((mygrids.kx[:mygrids.nxmid-1], mygrids.kx[mygrids.nxmid:]))

    tmp_pdf_id = 1
    pdflist = []
    tmp_pdf_id_fromSum = 1
    pdflist_fromSum = []

    plt.semilogy(kxnozero, phi2z[-1,:]/phi2z[0,:], linewidth=1, marker='o', color=gplot.mybluestd)
    plt.xlabel('$k_x$')
    plt.ylabel('$\\vert\\varphi_Z\\vert^2(t)/\\vert\\varphi_Z\\vert^2(0)$')
    plt.grid(True)
    plt.savefig(run.out_dir+'phi2zonal_vs_kx_last_timestep'+'_'+run.fnames[ifile]+'.pdf')
