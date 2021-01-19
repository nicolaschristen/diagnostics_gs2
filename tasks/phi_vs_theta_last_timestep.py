import pyfilm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy as cp
import gs2_plotting as gplot


def my_task_single(ifile, run, myin, myout, mygrids, mytime):

    iky_to_plot = [0,1,int(round(0.4/mygrids.ky[1]))]

    for iky in iky_to_plot:

        phi = np.squeeze(myout['phi'][iky,:,:,:])
        phi = phi[..., 0] + 1j*phi[..., 1]
        phi2 = np.abs(phi)**2

        # Reorder kx to be increasing, leaving out kx=0
        phi2 = np.concatenate((phi2[mygrids.nxmid:,:], phi2[1:mygrids.nxmid,:]), axis=0)

        # kx grid excluding zero
        kxnozero = np.concatenate((mygrids.kx[:mygrids.nxmid-1], mygrids.kx[mygrids.nxmid:]))

        # theta grid
        theta = myout['theta']

        tmp_pdf_id = 1
        pdflist = []
        tmp_pdf_id_fromSum = 1
        pdflist_fromSum = []

        for ix in range(kxnozero.size):

            plt.plot(theta, phi2[ix,:], linewidth=1, marker='o')
            plt.xlabel('$\\theta$')
            plt.ylabel('$\\vert\\varphi\\vert^2$')
            plt.title('$k_x=$'+str(round(kxnozero[ix],2))+' $k_y=$'+str(round(mygrids.ky[iky],2)))
            plt.grid(True)

            tmp_pdfname = 'tmp' + str(tmp_pdf_id)
            plt.savefig(run.out_dir+tmp_pdfname+'_'+run.fnames[ifile]+'.pdf')
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
            plt.cla()
            plt.clf()

        # Merge pdfs and save
        merged_pdfname = 'phi_ky_'+str(round(mygrids.ky[iky],2))+'_vs_theta_last_timestep'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
