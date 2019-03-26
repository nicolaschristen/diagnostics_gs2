from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as anim
import numpy as np
from math import pi
from math import ceil
import imageio
import os
import sys
#TESTpickle
#import scipy.io
import pickle
from PyPDF2 import PdfFileMerger, PdfFileReader
import scipy.interpolate as scinterp
import scipy.optimize as opt

import gs2_plotting as gplots

################################################################################
# A twist and shift chain is identified by it, iky and dmid = 0, ..., min(jtwist-1, (nakx-1)//2)
# which is the number of dkx between kx=0 and the smallest kx>0 that is a member of the chain.
################################################################################

################################################################################
# main
################################################################################

def my_task_single(ifile, run, myin, myout, task_space):  

    #
    # User parameters
    #
    
    # select chains
    naky = 7
    iky_list = [i for i in range(1,naky)] # negative means all nonzero ky
    if iky_list==[-1]:
        iky_list = [i for i in range(1,myout['ky'].size)]
    my_dmid = 0 # we include kxbar=my_dmid*dkx in our chain

    # Select time index for plot of phi vs ballooning angle
    #my_it = [10*i+3000 for i in range(6)]
    my_it = [-1]

    # make movies of phi and growthrate along ballooning angle ?
    make_movies = False

    #
    #
    #

    print('\n\n----------------------------------------')
    print('Starting single task : Floquet ...\n')

    my_vars = []

    if run.only_plot:
        
        my_vars = read_from_dat(ifile, run)

    else:
        
        my_vars = process_and_save_to_dat(ifile, run, myin, myout, my_dmid, iky_list)
        
    store_for_task_scan(my_vars, task_space)

    if not run.no_plot:

        plot_task_single(ifile, run, my_vars, my_it, my_dmid, make_movies)


    print('\n... single task completed.')


def process_and_save_to_dat(ifile, run, myin, myout, my_dmid, iky_list):

    plot_against_theta0_star = True # NDCPARAM

    t = myout['t']
    delt = myin['knobs']['delt']
    print('Time step size : ' + str(delt))
    nt = t.size
    nwrite = myin['gs2_diagnostics_knobs']['nwrite']
    print('nwrite : ' + str(nwrite))

    theta = myout['theta']
    ntheta = theta.size
    theta0 = myout['theta0']
    
    g_exb = myin['dist_fn_knobs']['g_exb']
    shat = myin['theta_grid_parameters']['shat']
    jtwist = int(myin['kt_grids_box_parameters']['jtwist'])
    
    # Floquet period
    if g_exb != 0.0:
        Tf = abs(2*pi*shat/g_exb)
    else:
        Tf = np.nan
    print('Floquet period : ' + str(Tf))
    # number of t-steps in Floquet period
    if g_exb != 0.0:
        Nf = int(round(Tf/delt))
        print('Number of t-steps in Floquet period : ' + str(Nf))
    else:
        Nf = np.nan

    kx_gs2 = myout['kx']
    ky = myout['ky']
    dky = 1./myin['kt_grids_box_parameters']['y0']
    dkx = 2.*pi*abs(shat)*dky/jtwist
    nakx = kx_gs2.size
    naky = ky.size
    ikx_max = int(round((nakx-1)/2))
    ikx_min = ikx_max+1
    
    # number of t-steps before ExB re-map
    if g_exb != 0.0:
        N = max(1, abs(int(round(dkx/(g_exb*delt*dky)))))
        print('Number of t-steps before ExB re-map : ' + str(N))
    else:
        N = np.nan

    phi2_gs2 = myout['phi2_by_mode'][:,:,:]

    phi_t_present = myout['phi_t_present']
    if phi_t_present:
        phi2_bytheta_gs2 = np.sum(np.power(myout['phi_t'],2), axis=4)
   
    # sorting kx_gs2 to get monotonic kx_bar
    kx_bar = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))
    phi2 = np.concatenate((phi2_gs2[:,:,ikx_min:], phi2_gs2[:,:,:ikx_min]), axis=2)
    if phi_t_present:
        phi2_bytheta = np.concatenate((phi2_bytheta_gs2[:,:,ikx_min:,:], phi2_bytheta_gs2[:,:,:ikx_min,:]), axis=2)

    # get kx and kx_star from kx_bar
    kx = np.zeros((nt,naky,nakx))
    kx_star = np.zeros((nt,naky,nakx))
    # @ it = 0, kx = kx_bar
    # First step is taken with 0.5*dt
    # Other steps taken with full dt
    for it in range(1,nt):
        for iky in range(naky):
            ikx_shift = int(round(g_exb*ky[iky]*delt*(nwrite*it-0.5)/dkx))
            for ikx in range(nakx):
                kx[it,iky,ikx] = kx_bar[ikx] + ikx_shift*dkx
                kx_star[it,iky,ikx] = kx[it,iky,ikx] - g_exb*ky[iky]*delt*(nwrite*it-0.5)
 
    # index of kx=0
    ikx0 = (nakx-1)//2
    
    #
    # In the following :
    # -- compute sum of phi2 along that chain and store in sum_phi2_chain[it]
    # -- compute ballooning angle and store in bloonang_chain[it][ibloon]
    # -- construct associated phi2 and store in phi2bloon_chain[it][ibloon]
    # -- compute associated growthrate gamma, such that :
    #        phinew = phi * exp(-i*omega*delt)
    #        gamma = Im(omega)
    #    and store in gamma_chain[it][ibloon]
    #
    
    ikx_members = []
    ikx_prevmembers = []
    bloonang = []
    phi2bloon = []
    phi2bloon_discont = []
    sum_phi2bloon = []
    max_phi2bloon = []
    gamma = []
    kx_star_for_gamma = []

    for iky in iky_list:

        ikx_members_chain = []
        ikx_prevmembers_chain = []
        bloonang_chain = []
        phi2bloon_chain = []
        phi2bloon_discont_chain = []
        sum_phi2bloon_chain = []
        max_phi2bloon_chain = []
        gamma_chain = []
        kx_star_for_gamma_chain = []

        iTf = 0
        gamma_floq = []
        kx_star_for_gamma_floq = []

        for it in range(nt):

            ikx_members_now = []
            ikx_prevmembers_now = []
            bloonang_now = []
            phi2bloon_now = []
            phi2bloon_discont_now = []
            sum_phi2bloon_now = 0

            # BLACK MAGIC LINE :
            # if the position of delt and it are swapped in the following multiplication,
            # the resulting ikx_shift can be different ! (e.g. it=297 for ~/gs2/flowtest/dkx_scan/dkx_2.in)
            if it==0:
                ikx_shift = 0
                ikx_shift_old = 0
            elif it==1:
                ikx_shift = int(round(g_exb*ky[iky]*delt*(nwrite*it-0.5)/dkx))
                ikx_shift_old = 0
            else:
                ikx_shift = int(round(g_exb*ky[iky]*delt*(nwrite*it-0.5)/dkx))
                ikx_shift_old = int(round(g_exb*ky[iky]*delt*(nwrite*(it-1)-0.5)/dkx))
    
            # Build collection of ikx's that are included
            # in the chain at time step it (ikx_members_now)
            # and at time step it-1 (ikx_prevmembers_now).

            # ikx such that our chain includes kxstar(t=0) = dmid*dkx
            ikx = ikx0 - ikx_shift + my_dmid
            ikxprev = ikx0 - ikx_shift_old + my_dmid

            while (ikx >= nakx): # moving from right to first connected kx within the set in GS2
                ikx = ikx-jtwist*iky
                ikxprev = ikxprev-jtwist*iky
            while (ikx >= 0):
                ikx_members_now.append(ikx)
                if ikxprev >= nakx:
                    ikx_prevmembers_now.append(np.nan)
                elif ikxprev < 0:
                    ikx_prevmembers_now.append(np.nan)
                else:
                    ikx_prevmembers_now.append(ikxprev)
                sum_phi2bloon_now = sum_phi2bloon_now + phi2[it,iky,ikx]                    
                ikx = ikx-jtwist*iky
                ikxprev = ikxprev-jtwist*iky

            ikx = ikx0 - ikx_shift + my_dmid + jtwist*iky
            ikxprev = ikx0 - ikx_shift_old + my_dmid + jtwist*iky
            while (ikx < 0): # moving from left to first connected kx within the set in GS2
                ikx = ikx+jtwist*iky
                ikxprev = ikxprev+jtwist*iky
            while (ikx < nakx):
                ikx_members_now.append(ikx)
                if ikxprev >= nakx:
                    ikx_prevmembers_now.append(np.nan)
                elif ikxprev < 0:
                    ikx_prevmembers_now.append(np.nan)
                else:
                    ikx_prevmembers_now.append(ikxprev)
                sum_phi2bloon_now = sum_phi2bloon_now + phi2[it,iky,ikx]
                ikx = ikx+jtwist*iky
                ikxprev = ikxprev+jtwist*iky

            # sort ikx of chain members in ascending order for time it
            # sort time it-1 accordingly
            idx_sort = sorted(range(len(ikx_members_now)), key=lambda k: ikx_members_now[k])
            ikx_members_now = [ikx_members_now[idx] for idx in idx_sort]
            ikx_prevmembers_now = [ikx_prevmembers_now[idx] for idx in idx_sort]

            if phi_t_present:
                
                # compute ballooning angle and construct associated phi2
                for imember in range(len(ikx_members_now)-1,-1,-1):
                    for itheta in range(ntheta):
                        if plot_against_theta0_star:
                            b_ang = theta[itheta] - kx_star[it,iky,ikx_members_now[imember]]/(shat*ky[iky])
                        else:
                            b_ang = theta[itheta] - kx[it,iky,ikx_members_now[imember]]/(shat*ky[iky])
                        bloonang_now.append(b_ang)
                        phi2bloon_now.append(phi2_bytheta[it,iky,ikx_members_now[imember],itheta])

                # Saving discontinuities to plot later
                for imember in range(len(ikx_members_now)-1,0,-1):
                    discont = abs(phi2_bytheta[it,iky,ikx_members_now[imember-1],0] \
                            - phi2_bytheta[it,iky,ikx_members_now[imember],-1])
                    phi2bloon_discont_now.append(discont)

                # Computing 'growthrate' for every kxstar present in the chain
                if it>0 and g_exb != 0.0:
                    # index of the Floquet oscillation we are in
                    iTf_new = int(round(delt*(it*nwrite-0.5)/Tf))
                    # If we enter the next Floquet oscillation,
                    # append gamma to gamma_chain
                    # and start working with new oscillation.
                    if iTf_new > iTf:
                        # Oh dare ! -- J. Bercow, 2019
                        idx_sort = [i[0] for i in sorted(enumerate(kx_star_for_gamma_floq), key=lambda x:x[1])]
                        kx_star_for_gamma_floq = [kx_star_for_gamma_floq[i] for i in idx_sort]
                        gamma_floq = [gamma_floq[i] for i in idx_sort]
                        # and append
                        gamma_chain.append(gamma_floq)
                        kx_star_for_gamma_chain.append(kx_star_for_gamma_floq)
                        gamma_floq = []
                        kx_star_for_gamma_floq = []
                        iTf = iTf_new
                    for imember in range(len(ikx_members_now)):
                        ikx = ikx_members_now[imember]
                        ikxprev = ikx_prevmembers_now[imember]
                        kx_star_for_gamma_floq.append(kx_star[it,iky,ikx])
                        if not np.isnan(ikxprev):
                            gam = 1./(2.*nwrite*delt) * np.log( \
                                    np.amax(phi2_bytheta[it,iky,ikx,:]) / np.amax(phi2_bytheta[it-1,iky,ikxprev,:]) )
                            gamma_floq.append(gam)
                        else:
                            gamma_floq.append(np.nan)

            ikx_members_chain.append(ikx_members_now)
            ikx_prevmembers_chain.append(ikx_prevmembers_now)
            bloonang_chain.append(bloonang_now)
            phi2bloon_chain.append(phi2bloon_now)
            phi2bloon_discont_chain.append(phi2bloon_discont_now)
            sum_phi2bloon_chain.append(sum_phi2bloon_now)
            max_phi2bloon_chain.append(max(phi2bloon_now))
    
        # Adding the chain to the full collection
        ikx_members.append(ikx_members_chain)
        ikx_prevmembers.append(ikx_prevmembers_chain)
        bloonang.append(bloonang_chain)
        phi2bloon.append(phi2bloon_chain)
        phi2bloon_discont.append(phi2bloon_discont_chain)
        sum_phi2bloon.append(sum_phi2bloon_chain)
        max_phi2bloon.append(max_phi2bloon_chain)
        if g_exb != 0.0:
            kx_star_for_gamma.append(kx_star_for_gamma_chain)
            gamma.append(gamma_chain)

    if g_exb==0.0:
        it_start = round(0.3*nt)
        gamma = np.zeros((nakx,len(iky_list)))
        tofit_sq = np.amax(phi2_bytheta,axis=3) # take the max in theta for each 2pi segment
        for ikx in range(nakx):
            for ichain in range(len(iky_list)):
                iky = iky_list[ichain]
                gam = get_growthrate(t,tofit_sq[:,iky,ikx],it_start)
                gam = gam/2. # because we fitted the square
                gamma[ikx,ichain] = gam

    # Saving variables to mat-file
    my_vars = {}
    my_vars['Nf'] = Nf
    my_vars['t'] = t
    my_vars['delt'] = delt
    my_vars['nwrite'] = nwrite
    my_vars['shat'] = shat
    my_vars['g_exb'] = g_exb
    my_vars['kx'] = kx
    my_vars['ky'] = ky
    my_vars['dkx'] = dkx
    my_vars['jtwist'] = jtwist
    my_vars['iky_list'] = iky_list
    my_vars['nakx'] = nakx
    my_vars['kx_bar'] = kx_bar
    my_vars['kx_star'] = kx_star
    my_vars['phi2'] = phi2
    my_vars['phi_t_present'] = phi_t_present
    my_vars['ikx_members'] = ikx_members
    my_vars['ikx_prevmembers'] = ikx_prevmembers
    my_vars['bloonang'] = bloonang
    my_vars['phi2bloon'] = phi2bloon
    my_vars['phi2bloon_discont'] = phi2bloon_discont
    my_vars['sum_phi2bloon'] = sum_phi2bloon
    my_vars['max_phi2bloon'] = max_phi2bloon
    my_vars['gamma'] = gamma
    my_vars['kx_star_for_gamma'] = kx_star_for_gamma

    #TESTpickle
    #mat_file_name = run.out_dir + run.fnames[ifile] + '.floquet.mat'
    #scipy.io.savemat(mat_file_name, my_vars)
    datfile_name = run.out_dir + run.fnames[ifile] + '.floquet.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(my_vars,outfile)

    return my_vars


################################################################################
# reading Floquet variables from mat-file
################################################################################

#TESTpickle
#def read_from_mat(ifile, run):
#
#    mat_file_name = run.out_dir + run.fnames[ifile] + '.floquet.mat'
#
#    my_vars = scipy.io.loadmat(mat_file_name, squeeze_me=True)
#
#    return my_vars
def read_from_dat(ifile, run):

    datfile_name = run.out_dir + run.fnames[ifile] + '.floquet.dat'

    with open(datfile_name, 'rb') as infile: # 'rb' stands for read bytes
        my_vars = pickle.load(infile)

    return my_vars


################################################################################
# plotting
################################################################################
    
def plot_task_single(ifile, run, my_vars, my_it, my_dmid, make_movies):
        
    Nf = my_vars['Nf']
    t = my_vars['t']
    delt = my_vars['delt']
    nwrite = my_vars['nwrite']
    shat = my_vars['shat']
    g_exb = my_vars['g_exb']
    kx = my_vars['kx']
    ky = my_vars['ky']
    dkx = my_vars['dkx']
    jtwist = my_vars['jtwist']
    iky_list = my_vars['iky_list']
    nakx = my_vars['nakx']
    kx_bar = my_vars['kx_bar']
    kx_star = my_vars['kx_star']
    phi2 = my_vars['phi2']
    phi_t_present = my_vars['phi_t_present']
    ikx_members = my_vars['ikx_members']
    ikx_prevmembers = my_vars['ikx_prevmembers']
    bloonang = my_vars['bloonang']
    phi2bloon = my_vars['phi2bloon']
    phi2bloon_discont = my_vars['phi2bloon_discont']
    sum_phi2bloon = my_vars['sum_phi2bloon']
    max_phi2bloon = my_vars['max_phi2bloon']
    kx_star_for_gamma = my_vars['kx_star_for_gamma']
    gamma = my_vars['gamma']
    
    Tf = Nf*delt
    nt = t.size

    myfig = plt.figure(figsize=(12,8))

    for ichain in range(len(iky_list)):
    
        iky = iky_list[ichain]

        # plot phi2 vs t for each kx
        plt.title('$k_y={:.2f}$'.format(ky[iky]))
        plt.xlabel('$$t\\ [r_r/v_{thr}]$$')
        my_ylabel = '$\\ln \\left(\\vert \\langle \\phi \\rangle_\\theta \\vert ^2\\right)$'
        plt.ylabel(my_ylabel)
        plt.grid(True)
        my_colorlist = plt.cm.plasma(np.linspace(0,1,kx_bar.size))
        my_legend = []
        kxs_to_plot=kx_bar
        for ikx in range(kx_bar.size):
            if kx_bar[ikx] in kxs_to_plot:
                plt.plot(t, np.log(phi2[:,1,ikx]), color=my_colorlist[ikx])
                my_legend.append('$\\rho_i\\bar{k}_x = '+str(kx_bar[ikx])+'$')
        plt.legend(my_legend)
        axes=plt.gca()

        pdfname = 'phi2_by_kx_iky_{:d}'.format(iky)
        pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
        plt.savefig(pdfname)
        
        plt.clf()
        plt.cla()

        if (phi_t_present):
           
            # plot phi2 of chosen chain vs ballooning angle at chosen time
            for it in my_it:
                plt.xlabel('$\\theta -\\theta_0^*$') # NDCPARAM: check plot_against_theta0_star
                plt.ylabel('$\\vert \\phi \\vert ^2$')
                plt.title('$t=$ '+str(t[it])+', $k_y={:.2f}$'.format(ky[iky]))
                plt.grid(True)
                plt.gca().set_xlim(np.min(bloonang[ichain][it]),np.max(bloonang[ichain][it]))
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
                plt.plot(bloonang[ichain][it], phi2bloon[ichain][it], marker='o', \
                        markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)

                pdfname = 'balloon_it_' + str(it) + '_iky_' + str(iky) + '_dmid_' + str(my_dmid)
                pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
                plt.savefig(pdfname)
                
                plt.clf()
                plt.cla()
            # NDCTEST: plotting mutliple times together
            #plt.xlabel('$\\theta -\\theta_0$')
            #plt.ylabel('$\\vert \\phi \\vert ^2$')
            #plt.grid(True)
            #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            #line2, = plt.plot(bloonang_chain[210], phi2bloon_chain[210], marker='None', linestyle='--', linewidth=4.0, color=gplots.oxbluel)
            #line3, = plt.plot(bloonang_chain[310], phi2bloon_chain[310], marker='None', linestyle=':', linewidth=3.0, color=gplots.oxbluell)
            #line1, = plt.plot(bloonang_chain[110], phi2bloon_chain[110], marker='None', linewidth=5.0, color=gplots.oxblue)
            #plt.legend([line1,line2,line3], ['$t=$ '+"{:.1f}".format(t[110]),'$t=$ '+"{:.1f}".format(t[210]),'$t=$ '+"{:.1f}".format(t[310])])
            #plt.savefig('two_times_floquet.pdf')
            #plt.clf()
            #plt.cla()
            # endNDCTEST

            # plot sum of phi2 along chain vs time
            plt.xlabel('$t$')
            plt.ylabel('$\\ln \\left(\\sum_{K_x}\\vert \\langle \\phi \\rangle_\\theta \\vert ^2\\right)$')
            plt.title('Sum along ballooning mode, $k_y={:.2f}$'.format(ky[iky]))
            plt.grid(True)
            plt.semilogy(t, sum_phi2bloon[ichain], color=gplots.myblue, linewidth=3.0) 
            pdfname = 'floquet_vs_t'+ '_iky_' + str(iky) + '_dmid_' + str(my_dmid) 
            pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
            plt.savefig(pdfname)
            
            plt.clf()
            plt.cla()

            # plot max of phi2 along chain vs time
            plt.xlabel('$t$')
            plt.ylabel('$\\max\\left(\\ln \\vert \\langle \\phi \\rangle_\\theta \\vert ^2\\right)$')
            plt.title('Max along ballooning mode, $k_y={:.2f}$'.format(ky[iky]))
            plt.grid(True)
            plt.semilogy(t, max_phi2bloon[ichain], color=gplots.myblue, linewidth=3.0) 
            pdfname = 'max_phi2_vs_t'+ '_iky_' + str(iky) + '_dmid_' + str(my_dmid) 
            pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
            plt.savefig(pdfname)
            
            plt.clf()
            plt.cla()
        
            # plot gamma vs kxstar/(pi*shat*ky)
            #plt.title('$k_y={:.2f}$'.format(ky[iky]))
            #plt.xlabel('$k_x^*/(\\pi\\hat{s}k_y)$')
            #plt.ylabel('$\\langle\\gamma\\rangle_\\theta$')
            #plt.grid(True)
            #plt.plot(kxstar_over_ky[ichain], gamma[ichain], color=gplots.myblue, linewidth=3.0, marker='o') 
            #pdfname = 'gamma_vs_kxstar_over_ky' + '_iky_' + str(iky) + '_dmid_' + str(my_dmid)
            #pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
            #plt.savefig(pdfname)
            #
            #plt.clf()
            #plt.cla()

            # make movie of phi2 vs ballooning angle over time
            if (make_movies):
                
                moviename = 'phi_bloon' + '_iky_' + str(iky) + '_dmid_' + str(my_dmid)
                moviename = run.out_dir + moviename + '_' + run.fnames[ifile] + '.mp4'

                max_it_for_mov = nt
                it_step_for_mov = 3
                # find global min and max of ballooning angle
                bloonang_min = 0.
                bloonang_max = 0.
                for it in range(max_it_for_mov):
                    if np.min(bloonang[ichain][it]) < bloonang_min:
                        bloonang_min = np.min(bloonang[ichain][it])
                    if np.max(bloonang[ichain][it]) > bloonang_max:
                        bloonang_max = np.max(bloonang[ichain][it])
               
                print("\ncreating movie of phi vs ballooning angle ...")
                xdata1, ydata1 = [], []
                l1, = plt.plot([],[], marker='o', color=gplots.myblue, \
                        markersize=12, markerfacecolor=gplots.myblue, markeredgecolor=gplots.myblue, linewidth=3.0)
                xdata2, ydata2 = [], []
                l2, = plt.plot([],[], linestyle='', \
                        marker='o', markersize=8, markerfacecolor='r', markeredgecolor='r')
                plt.xlabel('$\\theta -\\theta_0^*$') # NDCPARAM: check for plot_against_theta0_star
                plt.ylabel('$\\vert \\phi \\vert ^2$')
                plt.grid(True)
                plt.gca().set_xlim(bloonang_min,bloonang_max)
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
                # Initialize lines
                def init_mov():
                    l1.set_data([], [])
                    l2.set_data([], [])
                    return l1,l2
                # Update lines
                def update_mov(it):
                    # Update chain
                    sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1)))
                    xdata1 = bloonang[ichain][it]
                    ydata1 = phi2bloon[ichain][it]
                    l1.set_data(xdata1,ydata1)
                    ymin = np.amin(phi2bloon[ichain][it])
                    ymax = np.amax(phi2bloon[ichain][it])
                    ax = plt.gca()
                    ax.set_ylim(ymin,ymax)
                    ax.set_title('$k_y={:.2f}, t={:.2f}$'.format(ky[iky],t[it]))
                    # Update discontinuities at 2pi interfaces
                    bloonang_bndry = []
                    for imember in range(len(ikx_members[ichain][it])-1,0,-1):
                        x_sep = pi-kx_star[it,iky,ikx_members[ichain][it][imember]]/(shat*ky[iky])
                        bloonang_bndry.append(x_sep)
                    xdata2 = bloonang_bndry
                    ydata2 = phi2bloon_discont[ichain][it]
                    l2.set_data(xdata2,ydata2)
                    return l1, l2

                mov = anim.FuncAnimation(myfig,update_mov,init_func=init_mov, \
                        frames=range(0,max_it_for_mov,it_step_for_mov),blit=False,interval=10)
                writer = anim.writers['ffmpeg'](fps=30,bitrate=-1,codec='libx264')
                mov.save(moviename,writer=writer,dpi=100)
                plt.clf()
                plt.cla()
                print("\n... movie completed.")

    # plot gamma vs (kxstar,ky), for every Floquet oscillation in the simulation
    if phi_t_present:

        cbarmax = 0.5
        cbarmin = -0.5

        if g_exb != 0.0:

            tmp_pdf_id = 1
            pdflist = []
            # Finer and regular kx, ky mesh for contour plot of gamma
            nakx_fine = (kx_bar.size-1)*1e4+1
            kx_grid_fine = np.linspace(np.amin(kx_bar)-dkx/2.,np.amax(kx_bar)+dkx/2.,nakx_fine)
            ky_grid_fine = np.zeros(len(iky_list))
            for ichain in range(len(iky_list)):
                iky = iky_list[ichain]
                ky_grid_fine[ichain] = ky[iky]

            iTfmax = len(gamma[0])-1

            for iTf in range(iTfmax,-1,-1):
                # First arrange kx,ky,gamma similarly to fine meshes above
                npoints = 0
                for ichain in range(len(iky_list)):
                    npoints = npoints + len(kx_star_for_gamma[ichain][iTf])
                kx_grid_1d = np.zeros(npoints)
                ky_grid_1d = np.zeros(npoints)
                gamma_1d = np.zeros(npoints)
                istart = 0
                for ichain in range(len(iky_list)):
                    iky = iky_list[ichain]
                    for ikxstar in range(len(kx_star_for_gamma[ichain][iTf])):
                        ipoint = istart + ikxstar
                        kx_grid_1d[ipoint] = kx_star_for_gamma[ichain][iTf][ikxstar]
                        ky_grid_1d[ipoint] = ky[iky]
                        gamma_1d[ipoint] = gamma[ichain][iTf][ikxstar]
                    istart = istart + len(kx_star_for_gamma[ichain][iTf])
                # then interpolate to nearest neighbour on fine, regular mesh
                gamma_fine = scinterp.griddata((kx_grid_1d,ky_grid_1d),gamma_1d, \
                        (kx_grid_fine[None,:],ky_grid_fine[:,None]),method='nearest')
                # Set the colorbar limits according to the last Floquet oscillation
                if iTf == iTfmax:
                    #cbarmax = np.nanmax(gamma_fine)
                    cbarmax = 0.5
                    #cbarmin = np.nanmin(gamma_fine)
                    cbarmin = -0.5
                # and plot
                if len(iky_list)>1: # many ky: plot contour
                    my_title = '$d\\log(\\varphi)/dt, N_F={:d}/{:d}$'.format(iTf+1,len(gamma[ichain]))
                    my_xlabel = '$k_x^*$'
                    my_ylabel = '$k_y$'
                    gplots.plot_2d(gamma_fine,kx_grid_fine,ky_grid_fine,cbarmin,cbarmax,
                            xlab=my_xlabel,ylab=my_ylabel,title=my_title,cmp='RdBu_r')
                else: # single ky: 1d plot vs kxstar
                    plt.plot(kx_grid_1d,gamma_1d,linewidth=3.0,color=gplots.myblue)
                    plt.xlabel('$k_x^*$')
                    plt.ylabel('$d\\log(\\varphi)/dt$')
                    plt.title('$k_y={:.2f}, N_F={:d}/{:d}$'.format(ky[iky_list[0]],iTf+1,len(gamma[ichain])))
                    ax = plt.gca()
                    ax.set_ylim(cbarmin,cbarmax)
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplots.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1
            pdflist = pdflist[::-1] # re-order since we iterated from last oscillation
            merged_pdfname = 'gamma_vs_kxky' + '_dmid_' + str(my_dmid)
            gplots.merge_pdfs(pdflist, merged_pdfname, run, ifile)

        else: # g_exb = 0.0

            if len(iky_list)>1: # many ky: plot contour
                ky_to_plot = np.zeros(len(iky_list))
                for ichain in len(iky_list):
                    ky_to_plot[ichain] = ky[iky_list[ichain]]
                my_title = '$d\\log(\\varphi)/dt$'
                my_xlabel = '$k_x^*$'
                my_ylabel = '$k_y$'
                gplots.plot_2d(gamma,kx_bar,ky_to_plot,cbarmin,cbarmax,
                        xlab=my_xlabel,ylab=my_ylabel,title=my_title,cmp='RdBu_r')
            else: # single ky: 1d plot vs kxstar
                plt.plot(kx_bar,gamma[:,-1],linewidth=3.0,color=gplots.myblue)
                plt.xlabel('$k_x^*$')
                plt.ylabel('$d\\log(\\varphi)/dt$')
                plt.title('$k_y={:.2f}'.format(ky[iky_list[0]]))
                ax = plt.gca()
                ax.set_ylim(cbarmin,cbarmax)

            pdfname = 'gamma_vs_kxky' + '_dmid_' + str(my_dmid)
            pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
            plt.savefig(pdfname)
            
            plt.clf()
            plt.cla()

    # Plot sum and max of phi2 vs time for every ky
    # Fit each curve and plot gamma_avg vs ky

    # Start comparing simulations at time-step it_start = N_start*Tfloquet/dt
    # ie after N_start Floquet oscillations
    # Normalise sum_phi2 by sum_phi2[it_start] for each run
    if g_exb != 0.0:
        N_start = 0
        it_start = int(round((N_start*Tf/delt)/nwrite))
    else:
        it_start = round(0.3*nt)

    t_collec = []
    sum_phi2_collec = []
    max_phi2_collec = []
    slope_sum = np.zeros(len(iky_list))
    slope_max = np.zeros(len(iky_list))
    
    for ichain in range(len(iky_list)):

        iky = iky_list[ichain]

        sum_phi2_tmp = np.zeros(len(sum_phi2bloon[ichain])-it_start)
        max_phi2_tmp = np.zeros(len(sum_phi2bloon[ichain])-it_start)
        for it in range(sum_phi2_tmp.size):
            sum_phi2_tmp[it] = sum_phi2bloon[ichain][it_start+it]
            max_phi2_tmp[it] = max_phi2bloon[ichain][it_start+it]
        sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
        sum_phi2_collec.append(sum_phi2_tmp)
        max_phi2_tmp = max_phi2_tmp/max_phi2_tmp[0]
        max_phi2_collec.append(max_phi2_tmp)
        
        t_tmp = np.zeros(len(t)-it_start)
        for it in range(t_tmp.size):
            t_tmp[it] = t[it_start+it]
        t_collec.append(t_tmp)

        [gam,dummy] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
        slope_sum[ichain] = gam/2. # divide by 2 because fitted square
        [gam,dummy] = leastsq_lin(t_tmp,np.log(max_phi2_tmp))
        slope_max[ichain] = gam/2. # divide by 2 because fitted square
        #gam = get_growthrate(t,sum_phi2bloon[ichain],it_start)
        #slope_sum[ichain] = gam/2. # divide by 2 because fitted square
        #gam = get_growthrate(t,max_phi2bloon[ichain],it_start)
        #slope_max[ichain] = gam/2. # divide by 2 because fitted square
    
    plt.figure(figsize=(12,8))
    plt.xlabel('$t$')
    plt.ylabel('$\\sum_{K_x}\\vert \\langle\\phi\\rangle_\\theta \\vert ^2$')
    #plt.title('Sum along a single ballooning mode')
    plt.grid(True)
    my_legend = []
    my_colorlist = plt.cm.YlOrBr(np.linspace(0.2,1,len(iky_list))) # for newalgo
    #my_colorlist = plt.cm.YlGnBu(np.linspace(0.2,1,len(iky_list))) # for oldalgo
    for ichain in range(len(iky_list)):
        my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[ichain]]))
        plt.semilogy(t_collec[ichain], sum_phi2_collec[ichain], color=my_colorlist[ichain], linewidth=3.0)
    plt.legend(my_legend)
    pdfname = 'floquet_sum_vs_t_all_ky' + '_dmid_' + str(my_dmid) 
    pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
    plt.savefig(pdfname)
    plt.clf()
    plt.cla()

    plt.figure(figsize=(12,8))
    plt.xlabel('$t$')
    plt.ylabel('$\\max_{K_x}\\vert \\langle\\phi\\rangle_\\theta \\vert ^2$')
    plt.grid(True)
    my_legend = []
    my_colorlist = plt.cm.YlOrBr(np.linspace(0.2,1,len(iky_list))) # for newalgo
    #my_colorlist = plt.cm.YlGnBu(np.linspace(0.2,1,len(iky_list))) # for oldalgo
    for ichain in range(len(iky_list)):
        my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[ichain]]))
        plt.semilogy(t_collec[ichain], max_phi2_collec[ichain], color=my_colorlist[ichain], linewidth=3.0)
    plt.legend(my_legend)
    pdfname = 'floquet_max_vs_t_all_ky' + '_dmid_' + str(my_dmid) 
    pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
    plt.savefig(pdfname)
    plt.clf()
    plt.cla()
    
    plt.figure(figsize=(12,8))
    plt.xlabel('$k_y$')
    plt.ylabel('$\\langle\\gamma\\rangle_t$')
    plt.grid(True)
    ky_to_plot = np.zeros(len(iky_list))
    for i in range(len(iky_list)):
        ky_to_plot[i] = ky[iky_list[i]]
    plt.plot(ky_to_plot, slope_sum, color=gplots.myblue, linewidth=3.0)
    pdfname = 'gamma_from_sum_vs_ky' + '_dmid_' + str(my_dmid) 
    pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
    plt.savefig(pdfname)
    plt.clf()
    plt.cla()
    
    plt.figure(figsize=(12,8))
    plt.xlabel('$k_y$')
    plt.ylabel('$\\langle\\gamma\\rangle_t$')
    plt.grid(True)
    ky_to_plot = np.zeros(len(iky_list))
    for i in range(len(iky_list)):
        ky_to_plot[i] = ky[iky_list[i]]
    plt.plot(ky_to_plot, slope_max, color=gplots.myblue, linewidth=3.0)
    pdfname = 'gamma_from_max_vs_ky' + '_dmid_' + str(my_dmid) 
    pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
    plt.savefig(pdfname)
    plt.clf()
    plt.cla()

    plt.close()


################################################################################
# storing quantities for scan-plots
################################################################################

def store_for_task_scan(my_vars, task_space):

    task_space.t = my_vars['t']
    task_space.delt = my_vars['delt']
    task_space.iky_list = my_vars['iky_list']
    task_space.nwrite = my_vars['nwrite']
    task_space.dkx = my_vars['dkx']
    task_space.Tf = my_vars['Nf']*my_vars['delt']
    task_space.sum_phi2bloon = my_vars['sum_phi2bloon']


################################################################################
# Part of task for scans
################################################################################

def task_scan(run, full_space):

    # Start comparing simulations at time-step it_start = N_start*Tfloquet/dt
    # ie after N_start Floquet oscillations
    # Normalise sum_phi2 by sum_phi2[it_start] for each run
    
    N_start = 2
    
    sum_phi2 = []
    t = []
    delt = np.zeros(len(run.fnames))
    dkx = np.zeros(len(run.fnames))
    slope = np.zeros(len(run.fnames))
    
    if not run.no_plot:

        for ichain in range(len(full_space[ifile]['floquet'].iky_list)):

            iky = iky_list[ichain]

            for ifile in range(len(run.fnames)):
                
                Tf = full_space[ifile]['floquet'].Tf
                delt[ifile] = full_space[ifile]['floquet'].delt
                dkx[ifile] = full_space[ifile]['floquet'].dkx
                nwrite = full_space[ifile]['floquet'].nwrite

                it_start = int(round((N_start*Tf/delt[ifile])/nwrite))

                sum_phi2_tmp = np.zeros(len(full_space[ifile]['floquet'].sum_phi2bloon[ichain])-it_start)
                for it in range(sum_phi2_tmp.size):
                    sum_phi2_tmp[it] = full_space[ifile]['floquet'].sum_phi2bloon[ichain][it_start+it]
                sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
                sum_phi2.append(sum_phi2_tmp)
                
                t_tmp = np.zeros(len(full_space[ifile]['floquet'].t)-it_start)
                for it in range(t_tmp.size):
                    t_tmp[it] = full_space[ifile]['floquet'].t[it_start+it]
                t.append(t_tmp)

                [a,dummy] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
                slope[ifile] = a
    
            print('Slopes:')
            print(slope)
            plt.figure(figsize=(12,8))
            
            # Plot phi2 summed along chain
            plt.xlabel('$t$')
            plt.ylabel('$\\ln \\left(\\sum_{K_x}\\vert \\langle\\phi\\rangle_\\theta \\vert ^2\\right)$')
            #plt.title('Sum along a single ballooning mode')
            plt.grid(True)
            my_legend = []
            my_colorlist = plt.cm.YlOrBr(np.linspace(0.2,1,len(run.fnames))) # for newalgo
            #my_colorlist = plt.cm.YlGnBu(np.linspace(0.2,1,len(run.fnames))) # for oldalgo
            for ifile in range(len(run.fnames)):
                #my_legend.append('$\\Delta t =$'+str(full_space[ifile]['floquet'].delt))
                my_legend.append('$\\Delta k_x = {:.3f}$'.format(full_space[ifile]['floquet'].dkx))
                plt.plot(t[ifile], np.log(sum_phi2[ifile]), color=my_colorlist[ifile], linewidth=3.0)
            plt.legend(my_legend)
            axes = plt.gca()
            axes.set_ylim([-0.5, 13.75])

            # Plot growthrates within phi2 plot
            subaxes = plt.axes([0.65, 0.25, 0.3, 0.25])
            subaxes.tick_params(labelsize=18)
            plt.xlabel('$\\Delta k_x$',fontsize=20)
            #plt.ylabel('$\\langle \\gamma \\rangle_t$',fontsize=20)
            plt.title('Time averaged growth-rate',fontsize=20)
            plt.grid(True)
            plt.plot(dkx, slope, marker='o', color='black', \
                    markersize=10, markerfacecolor='none', markeredgecolor='black', linewidth=2.0)
            gamma_converged = 0.06418364
            #plt.axhline(y=gamma_converged, color='grey', linestyle='--', linewidth=2.0) # for oldalgo
            subaxes.set_ylim([0.06, 0.13])
            
            gplots.save_plot(run.scan_name, run)
            
            plt.clf()
            plt.cla()

        plt.close()

def leastsq_lin(x, y):
    
    # y_fit = a*x + b
    # minimising sum((y - f_fit)^2)
    N_x = x.size

    a = 1./(N_x*np.sum(np.power(x,2)) - np.sum(x)**2) * (N_x*np.sum(np.multiply(x,y)) - np.sum(x)*np.sum(y))
    
    b = 1./(N_x*np.sum(np.power(x,2)) - np.sum(x)**2) * (-1.*np.sum(x)*np.sum(np.multiply(x,y)) + np.sum(np.power(x,2))*np.sum(y))

    return [a, b]

def get_growthrate(t,tofit,it_start):
   
    popt, pcov = opt.curve_fit(lin_func, t[it_start:], np.log(tofit[it_start:]))
    return popt[0]

def lin_func(x,a,b):
    return a*x+b
