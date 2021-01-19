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

    for ix in range(kxnozero.size):

        plt.semilogy(mytime.time, phi2z[:,ix]/phi2z[0,ix], linewidth=1)
        plt.xlabel('$t$')
        plt.ylabel('$\\vert\\varphi_Z\\vert^2(t)/\\vert\\varphi_Z\\vert^2(0)$')
        plt.title('$kx=$'+str(round(kxnozero[ix],2)))
        plt.grid(True)

        tmp_pdfname = 'tmp' + str(tmp_pdf_id)
        plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        plt.cla()
        plt.clf()

    # Merge pdfs and save
    merged_pdfname = 'phi2zonal_vs_t'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

