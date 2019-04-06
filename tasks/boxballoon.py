################################################################################
# A twist and shift chain is identified by it, iky and dmid = 0, ..., min(jtwist-1, (nakx-1)//2)
# which is the number of dkx between kx=0 and the smallest kx>0 that is a member of the chain.
################################################################################

################################################################################
# Task to write/plot several quantities for linear simulations done in box mode, as a ballooning chain in ballooning angle theta.
################################################################################

import numpy as np
from math import pi
import scipy.optimize as opt
from matplotlib import pyplot as plt
import gs2_plotting as gplot
import pickle
import matplotlib
import gs2_plotting as gplot
import imageio
import os
import sys
def my_task_single(ifile, run, myin, myout):    

    # User parameters
    dump_at_start = 0.3 # fraction of initial time to dump when fitting
    
    # Name for data file to be written to/read from.
    datfile_name = run.out_dir + run.fnames[ifile] + '.boxballoon.dat'
    print(datfile_name) 
    # Compute and save growthrate
    if not run.only_plot:

        grid_option = myin['kt_grids_knobs']['grid_option']
        if not 'box' == grid_option:
            print('Error - grid_option should be "box"')
            quit()
        
        t = myout['t']
        nt = t.size
        it_start = round(nt*dump_at_start)
    
        shat = myin['theta_grid_parameters']['shat']
        jtwist = int(myin['kt_grids_box_parameters']['jtwist'])
        
        theta = myout['theta']
        ntheta = len(theta)
        kx_gs2 = myout['kx']
        ky = myout['ky']
        nakx = kx_gs2.size
        naky = ky.size

        theta0_gs2 = myout['theta0']
        theta0 = np.zeros(theta0_gs2.shape)

        gbdrift = myout['gbdrift']
        gbdrift0 = myout['gbdrift0']
        cvdrift = myout['cvdrift']
        cvdrift0 = myout['cvdrift0']

        gds2 = myout['gds2']
        gds21 = myout['gds21']
        gds22 = myout['gds22']
        
        dky = 1./myin['kt_grids_box_parameters']['y0']
        dkx = 2.*pi*abs(shat)*dky/jtwist
        phi_gs2 = myout['phi']      # Potential [ky,kx,theta,ri]
        phi2_gs2 = np.sqrt(phi_gs2[:,:,:,0]**2 + phi_gs2[:,:,:,1]**2)   # Potential squared [ky,kx,theta]
        theta = myout['theta']
        # Rearrange kx so that it is monotonically increasing     
        ikx_max = int(round((nakx-1)/2))
        ikx_min = ikx_max+1
        for i in range(len(theta0_gs2)):
            theta0[i] = np.concatenate((theta0_gs2[i,ikx_min:],theta0_gs2[i,:ikx_min]))
        kx = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))
        phi2 = np.concatenate((phi2_gs2[:,ikx_min:,:], phi2_gs2[:,:ikx_min,:]), axis=1)
        # Lines to trim array of NaNs from running a linear sim for too long.
        #for it in range(nt):
        #    for ikx in range(nakx):
        #        for iky in range(naky):
        #            if it < it_end and not np.isfinite(phi2[it,iky,ikx]):
        #                it_end = it - 10 
        #                break
        
        # This is now the kx=0 index.
        ikx0 = (nakx-1)//2
        
        # ikx that we want to be at the middle of the ballooning chain (basically theta0?)
        # For now this is ikx0, i.e. kx = 0 at middle of chain.

        bloontheta = []
        bloonphi2 = []
        bloonikxs = []
        bloonkxs = []
        bloonkperps = []
        bloondrifts = []
        # Construct a ballooning chain for each ky (Excluding ky=0 where iky = 0, so there is no ballooning chain)
        for iky in range(1,naky):
            bloontheta.append([])
            bloonphi2.append([])
            bloonikxs.append([])
            bloonkxs.append([])
            bloonkperps.append([])
            bloondrifts.append([])
            # Can have several ballooning chains, each with different theta0.
            # The number of ballooning chains is determined by iky*jtwist
            it = 0
            for itheta0 in range(iky*jtwist):
                print(itheta0)
                bloontheta[iky-1].append([])
                bloonphi2[iky-1].append([])
                bloonikxs[iky-1].append([])
                bloonkxs[iky-1].append([])
                bloonkperps[iky-1].append([])
                bloondrifts[iky-1].append([])
                # for a ballooning chain indexed by N and iky, where N=0 at itheta0, Nmin <= 0 and Nmax >= 0 are the min and max indices:
                Nmin = -((itheta0+ikx0)//(iky*jtwist))
                Nmax = (nakx-(ikx0+1+itheta0))//(iky*jtwist)
                # Make array of indices of kx that contribute to the chain.
                ikxs = list(np.linspace(ikx0 + itheta0 + iky*jtwist*Nmin, ikx0 + itheta0 + iky*jtwist*Nmax, Nmax-Nmin+1, dtype = int))
                kxs = kx[ikxs]
                theta0s = theta0[iky,ikxs]
                bloonikxs[iky-1][it].extend(ikxs)
                bloonkxs[iky-1][it].extend(kxs)
                if shat>0.:
                    Ncounter = Nmax
                else:
                    Ncounter = Nmin
                for iikx in range(len(ikxs)):
                    bloontheta[iky-1][it].append([])
                    bloonphi2[iky-1][it].append([])
                    bloonkperps[iky-1][it].append([])
                    bloondrifts[iky-1][it].append([])
                    for itheta in range(ntheta):
                        bloontheta[iky-1][it][iikx].append( theta[itheta] + 2*pi*Ncounter)
                        bloonphi2[iky-1][it][iikx].append( phi2[iky,ikxs[iikx],itheta] )
                        bloonkperps[iky-1][it][iikx].append( ky[iky]**2 * (gds2[itheta] + 2*theta0s[iikx]*gds21[itheta] + theta0s[iikx]**2 * gds22[itheta]) )
                        bloondrifts[iky-1][it][iikx].append( ky[iky] * (gbdrift[itheta] + cvdrift[itheta] + shat*theta0s[iikx]*(gbdrift0[itheta] + cvdrift0[itheta])) )
                    Ncounter -= np.sign(shat)
                it += 1  
        
        my_vars = {}
        my_vars['kx'] = kx
        my_vars['ky'] = ky
        my_vars['bloontheta'] = bloontheta
        my_vars['bloonphi2'] = bloonphi2
        my_vars['bloonikxs'] = bloonikxs
        my_vars['bloonkxs'] = bloonkxs
        my_vars['bloonkperps'] = bloonkperps
        my_vars['bloondrifts'] = bloondrifts
        with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
            pickle.dump(my_vars,outfile)

    # Otherwise, only_plot == True and we should read in my_vars so it is available in memory for plotting.
    else:
        with open(datfile_name, 'rb') as datfile:
            my_vars = pickle.load(datfile)

    if not run.no_plot:
        plot(ifile, run, my_vars)
     
def plot(ifile, run, my_vars):
    bloontheta = my_vars['bloontheta']
    bloonphi2 = my_vars['bloonphi2']
    bloonikxs = my_vars['bloonikxs']
    bloonkxs = my_vars['bloonkxs']
    bloonkperps = my_vars['bloonkperps']
    bloondrifts = my_vars['bloondrifts']
    kx = my_vars['kx'] 
    ky = my_vars['ky']

    tmp_pdf_id = 1
    pdflist = []
    if bloontheta is not None and bloonphi2 is not None:
        for iky in range(1,len(ky)):
            # Plot ballooning chain for each ky. Start with just a single ky for testing.
            it = 0
            if len(bloonikxs[iky-1]) > 1:
                dbloonkx = bloonkxs[iky-1][it][1]-bloonkxs[iky-1][it][0]
                dbloonikx = bloonikxs[iky-1][it][1]-bloonikxs[iky-1][it][0]
                bloonkxs[iky-1][it].append(bloonkxs[iky-1][it][-1] + dbloonkx)
                cmap = matplotlib.cm.get_cmap('nipy_spectral',len(bloonkxs[iky-1][it])-1)
                print(bloonkxs[iky-1][it])
                norm = matplotlib.colors.BoundaryNorm(np.array(bloonkxs[iky-1][it])-dbloonkx/2.0,len(bloonkxs[iky-1][it]))
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                # Make dummie mappable
                dummy_cax = plt.scatter(bloonkxs[iky-1][it], bloonkxs[iky-1][it], c=bloonkxs[iky-1][it], cmap=cmap)
                # Clear axis
                plt.cla()
                title = '$|\phi|^2$ for ballooning chain with $k_y={:1.2f}$'.format(ky[iky])
                del bloonkxs[iky-1][it][-1]
            else:
                cmap = matplotlib.cm.get_cmap('nipy_spectral',2)
                title = '$|\phi|^2$ for ballooning chain with $k_y={:1.2f}$'.format(ky[iky]) + ' and $k_x = {:1.2f}$'.format(bloonkxs[iky-1][it][0])
            fig = plt.figure(figsize=(12,8))
            for iikx in range(len(bloonikxs[iky-1][it])):
                plt.gca().plot(bloontheta[iky-1][it][iikx], bloonphi2[iky-1][it][iikx]/np.max(np.max(bloonphi2[iky-1][it])), color = cmap(iikx))
                plt.gca().plot(bloontheta[iky-1][it][iikx], bloonkperps[iky-1][it][iikx]/np.max(np.max(bloonkperps[iky-1][it])), color = cmap(iikx), linestyle='dashed')
                plt.gca().plot(bloontheta[iky-1][it][iikx], bloondrifts[iky-1][it][iikx]/np.max(np.max(bloondrifts[iky-1][it])), color = cmap(iikx), linestyle=':')
            plt.title(title)
            plt.grid()
            plt.xlabel('$\\theta$')
            if len(bloonikxs[iky-1]) > 1:
                cbar_ax = fig.add_axes([1.0, 0.1, 0.03, 0.8])
                if len(bloonikxs[iky-1]) > 25:
                    ticks = bloonkxs[iky-1][it][::2]
                else:
                    ticks = bloonkxs[iky-1][it]
                skipxlabel = len(bloonkxs[iky-1]) // 10
                if skipxlabel > 1:
                    for label in plt.gca().xaxis.get_ticklabels()[::skipxlabel]:
                        label.set_visible(False) 
            
                fig.colorbar(sm, cax = cbar_ax, ticks=ticks, format='%.1f', orientation='vertical')
                cbar_ax.set_title('$k_x$')
            tmp_pdfname = 'tmp'+str(tmp_pdf_id)
            gplot.save_plot(tmp_pdfname, run, ifile)
            pdflist.append(tmp_pdfname)
            tmp_pdf_id = tmp_pdf_id+1
        merged_pdfname = 'box_ballooning'
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

