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
    naky = (myin['kt_grids_box_parameters']['ny']-1)//3 + 1
    iky_list = [i for i in range(1,naky)] # [-1] means all nonzero ky
    #iky_list = [1] # NDCTEST
    if iky_list==[-1]:
        iky_list = [i for i in range(1,myout['ky'].size)]
    my_dmid = [i for i in range(5)] # we include kxbar=my_dmid*dkx in our chain

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
        print('Floquet period : ' + str(Tf))
    else:
        Tf = np.nan
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
    omega_gs2 = myout['omega_average'][:,:,:,0] # real frequency

    phi_t_present = myout['phi_t_present']
    if phi_t_present:
        phi2_bytheta_gs2 = np.sum(np.power(myout['phi_t'],2), axis=4)
   
    # sorting kx_gs2 to get monotonic kx_bar
    kx_bar = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))
    phi2 = np.concatenate((phi2_gs2[:,:,ikx_min:], phi2_gs2[:,:,:ikx_min]), axis=2)
    omega = np.concatenate((omega_gs2[:,:,ikx_min:], omega_gs2[:,:,:ikx_min]), axis=2)
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
    bloonang_bndry = []
    phi2bloon = []
    phi2bloon_discont = []
    phi2bloon_jump = []
    sum_phi2bloon = []
    max_phi2bloon = []
    omegabloon = []
    kxbarbloon = []
    gamma = []
    kx_star_for_gamma = []

    for iky in iky_list:
    
        ikx_members_ky = []
        ikx_prevmembers_ky = []
        bloonang_ky = []
        bloonang_bndry_ky = []
        phi2bloon_ky = []
        phi2bloon_discont_ky = []
        phi2bloon_jump_ky = []
        sum_phi2bloon_ky = []
        max_phi2bloon_ky = []
        omegabloon_ky = []
        kxbarbloon_ky = []
        gamma_ky = []
        kx_star_for_gamma_ky = []

        for dmid in my_dmid:

            ikx_members_chain = []
            ikx_prevmembers_chain = []
            bloonang_chain = []
            bloonang_bndry_chain = []
            phi2bloon_chain = []
            phi2bloon_discont_chain = []
            phi2bloon_jump_chain = []
            sum_phi2bloon_chain = []
            max_phi2bloon_chain = []
            gamma_chain = []
            omegabloon_chain = []
            kxbarbloon_chain = []
            kx_star_for_gamma_chain = []

            iTf = 0
            gamma_floq = []
            kx_star_for_gamma_floq = []

            for it in range(nt):

                ikx_members_now = []
                ikx_prevmembers_now = []
                bloonang_now = []
                bloonang_bndry_now = []
                phi2bloon_now = []
                phi2bloon_discont_now = []
                phi2bloon_jump_now = []
                sum_phi2bloon_now = 0
                omegabloon_now = []
                kxbarbloon_now = []

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
                ikx = ikx0 - ikx_shift + dmid
                ikxprev = ikx0 - ikx_shift_old + dmid

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

                ikx = ikx0 - ikx_shift + dmid + jtwist*iky
                ikxprev = ikx0 - ikx_shift_old + dmid + jtwist*iky
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

                # sort ikx of chain members at time it in left-to-right order (shat>0: descending, shat<0: ascending)
                # sort time it-1 accordingly
                idx_sort = sorted(range(len(ikx_members_now)), key=lambda k: ikx_members_now[k],reverse=(shat>0.))
                ikx_members_now = [ikx_members_now[idx] for idx in idx_sort]
                ikx_prevmembers_now = [ikx_prevmembers_now[idx] for idx in idx_sort]

                if phi_t_present:

                    member_range = range(len(ikx_members_now))
                    
                    # compute ballooning angle and construct associated phi2
                    for imember in member_range:
                        for itheta in range(ntheta):
                            if plot_against_theta0_star:
                                b_ang = theta[itheta] - kx_star[it,iky,ikx_members_now[imember]]/(shat*ky[iky])
                            else:
                                b_ang = theta[itheta] - kx[it,iky,ikx_members_now[imember]]/(shat*ky[iky])
                            bloonang_now.append(b_ang)
                            phi2bloon_now.append(phi2_bytheta[it,iky,ikx_members_now[imember],itheta])
                    
                    # construct chain of real frequency
                    for imember in member_range:
                        omegabloon_now.append(omega[it,iky,ikx_members_now[imember]])
                        kxbarbloon_now.append(kx_bar[ikx_members_now[imember]])

                    # Saving discontinuities and bloonang at link position to plot later
                    member_range = range(len(ikx_members_now)-1)
                    for imember in member_range:
                        phi2_l = phi2_bytheta[it,iky,ikx_members_now[imember],-1]
                        phi2_r = phi2_bytheta[it,iky,ikx_members_now[imember+1],0]
                        discont = abs(phi2_r-phi2_l)
                        phi2bloon_discont_now.append(discont)
                        jump = abs((phi2_r-phi2_l)/max(phi2_l,phi2_r))
                        phi2bloon_jump_now.append(jump)
                        b_ang_bndry = pi-kx_star[it,iky,ikx_members_now[imember]]/(shat*ky[iky])
                        bloonang_bndry_now.append(b_ang_bndry)

                ikx_members_chain.append(ikx_members_now)
                ikx_prevmembers_chain.append(ikx_prevmembers_now)
                bloonang_chain.append(bloonang_now)
                bloonang_bndry_chain.append(bloonang_bndry_now)
                phi2bloon_chain.append(phi2bloon_now)
                phi2bloon_discont_chain.append(phi2bloon_discont_now)
                phi2bloon_jump_chain.append(phi2bloon_jump_now)
                sum_phi2bloon_chain.append(sum_phi2bloon_now)
                omegabloon_chain.append(omegabloon_now)
                kxbarbloon_chain.append(kxbarbloon_now)
                max_phi2bloon_chain.append(max(phi2bloon_now))
    
            # Adding the chain to the collection for the current ky
            ikx_members_ky.append(ikx_members_chain)
            ikx_prevmembers_ky.append(ikx_prevmembers_chain)
            bloonang_ky.append(bloonang_chain)
            bloonang_bndry_ky.append(bloonang_bndry_chain)
            phi2bloon_ky.append(phi2bloon_chain)
            phi2bloon_discont_ky.append(phi2bloon_discont_chain)
            phi2bloon_jump_ky.append(phi2bloon_jump_chain)
            sum_phi2bloon_ky.append(sum_phi2bloon_chain)
            max_phi2bloon_ky.append(max_phi2bloon_chain)
            omegabloon_ky.append(omegabloon_chain)
            kxbarbloon_ky.append(kxbarbloon_chain)
    
        # Adding all chains with the current ky to the full collection
        ikx_members.append(ikx_members_ky)
        ikx_prevmembers.append(ikx_prevmembers_ky)
        bloonang.append(bloonang_ky)
        bloonang_bndry.append(bloonang_bndry_ky)
        phi2bloon.append(phi2bloon_ky)
        phi2bloon_discont.append(phi2bloon_discont_ky)
        phi2bloon_jump.append(phi2bloon_jump_ky)
        sum_phi2bloon.append(sum_phi2bloon_ky)
        max_phi2bloon.append(max_phi2bloon_ky)
        omegabloon.append(omegabloon_ky)
        kxbarbloon.append(kxbarbloon_ky)

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
    my_vars['dmid'] = my_dmid
    my_vars['nakx'] = nakx
    my_vars['kx_bar'] = kx_bar
    my_vars['kx_star'] = kx_star
    my_vars['phi2'] = phi2
    my_vars['phi_t_present'] = phi_t_present
    my_vars['ikx_members'] = ikx_members
    my_vars['ikx_prevmembers'] = ikx_prevmembers
    my_vars['bloonang'] = bloonang
    my_vars['bloonang_bndry'] = bloonang_bndry
    my_vars['phi2bloon'] = phi2bloon
    my_vars['phi2bloon_discont'] = phi2bloon_discont
    my_vars['phi2bloon_jump'] = phi2bloon_jump
    my_vars['sum_phi2bloon'] = sum_phi2bloon
    my_vars['max_phi2bloon'] = max_phi2bloon
    my_vars['omegabloon'] = omegabloon
    my_vars['kxbarbloon'] = kxbarbloon

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
    bloonang_bndry = my_vars['bloonang_bndry']
    phi2bloon = my_vars['phi2bloon']
    phi2bloon_discont = my_vars['phi2bloon_discont']
    phi2bloon_jump = my_vars['phi2bloon_jump']
    sum_phi2bloon = my_vars['sum_phi2bloon']
    max_phi2bloon = my_vars['max_phi2bloon']
    omegabloon = my_vars['omegabloon']
    kxbarbloon = my_vars['kxbarbloon']
    
    Tf = Nf*delt
    nt = t.size

    myfig = plt.figure(figsize=(12,8))

    # Plot sum and max of phi2 vs time for every ky
    # Fit each curve and plot gamma_avg vs ky

    # Start comparing simulations at time-step it_start = N_start*Tfloquet/dt
    # ie after N_start Floquet oscillations
    # Normalise sum_phi2 by sum_phi2[it_start] for each run
    skip_init = False # Start plotting at it=it_start, instead of it=0
    if g_exb != 0.0:
        N_start = 2 #30 # adapt this
        it_start = int(round((N_start*Tf/delt)/nwrite))
    else:
        fac = 0.5 # adapt this
        it_start = round(fac*nt) # adapt this

    t_collec = []
    sum_phi2_collec = []
    max_phi2_collec = []
    
    # Compute <gamma>_t
    slope_sum = np.zeros((len(iky_list),len(my_dmid)))
    slope_max = np.zeros((len(iky_list),len(my_dmid)))
    offset_max = np.zeros((len(iky_list),len(my_dmid)))
    for iiky in range(len(iky_list)):
        for idmid in range(len(my_dmid)):
            sum_phi2_tmp = np.zeros(len(sum_phi2bloon[iiky][idmid])-it_start)
            max_phi2_tmp = np.zeros(len(sum_phi2bloon[iiky][idmid])-it_start)
            for it in range(sum_phi2_tmp.size):
                sum_phi2_tmp[it] = sum_phi2bloon[iiky][idmid][it_start+it]
                max_phi2_tmp[it] = max_phi2bloon[iiky][idmid][it_start+it]
            if it_start > 0:
                sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
                max_phi2_tmp = max_phi2_tmp/max_phi2_tmp[0]
            sum_phi2_collec.append(sum_phi2_tmp)
            max_phi2_collec.append(max_phi2_tmp)
            
            t_tmp = np.zeros(len(t)-it_start)
            for it in range(t_tmp.size):
                t_tmp[it] = t[it_start+it]
            t_collec.append(t_tmp)

            [gam,dummy] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
            slope_sum[iiky][idmid] = gam/2. # divide by 2 because fitted square
            [gam,offset] = leastsq_lin(t_tmp,np.log(max_phi2_tmp))
            slope_max[iiky][idmid] = gam/2. # divide by 2 because fitted square
            offset_max[iiky][idmid] = offset
    # At this point:
    # fit_avg(t) ~ phi2(tstart) * exp[2*gam*t + offset]

    # Compute gamma_max from ln(phi2_max)
    if g_exb != 0.0:
        it_gamma_max = np.zeros((len(iky_list),len(my_dmid)))
        gamma_max = np.zeros((len(iky_list),len(my_dmid)))
        for iiky in range(len(iky_list)):
            for idmid in range(len(my_dmid)):
                # Start looking for derivatives one Floquet period before last time-step
                it_start_last_floq = max(int(len(max_phi2bloon[iiky][idmid]) - Tf//(delt*nwrite) - 1),0)
                it_end_last_floq = int(len(max_phi2bloon[iiky][idmid]) - 1)
                for it in range(it_start_last_floq, it_end_last_floq):
                    # Factor of 0.5 because we fit phi^2
                    gamma_max_tmp = 0.5 * 1./(2*delt*nwrite) * \
                            ( np.log(max_phi2bloon[iiky][idmid][it+1]) - np.log(max_phi2bloon[iiky][idmid][it-1]) )
                    if (gamma_max_tmp > gamma_max[iiky][idmid]):
                        it_gamma_max[iiky][idmid] = it
                        gamma_max[iiky][idmid] = gamma_max_tmp
        it_gamma_max = it_gamma_max.astype(int)
        # At this point:
        # fit_max(t) ~ phi2(tstart) * exp[2*gamma_max*(t-t_gamma_max)]

        # Save growthrates to dat-file
        my_vars = {}
        my_vars['ky'] = ky[1:]
        my_vars['gamma_avg'] = slope_max
        my_vars['gamma_max'] = gamma_max
        datfile_name = run.out_dir + run.fnames[ifile] + '.flowshear_lingrowth.dat'
        with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
            pickle.dump(my_vars,outfile)
    
    for idmid in range(len(my_dmid)):

        plt.figure(figsize=(12,8))
        plt.xlabel('$t$')
        plt.ylabel('$\\log\\sum_{K_x}\\vert \\langle\\phi\\rangle_\\theta \\vert ^2$')
        plt.grid(True)
        my_legend = []
        my_colorlist = plt.cm.YlOrBr(np.linspace(0.2,1,len(iky_list))) # for newalgo
        #my_colorlist = plt.cm.YlGnBu(np.linspace(0.2,1,len(iky_list))) # for oldalgo
        if skip_init:
            for iiky in range(len(iky_list)):
                my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[iiky]]))
                #plt.semilogy(t_collec[iiky], sum_phi2_collec[iiky], color=my_colorlist[iiky], linewidth=3.0)
                plt.plot(t_collec[iiky][idmid], np.log(sum_phi2_collec[iiky][idmid]), color=my_colorlist[iiky], linewidth=3.0)
        else:
            for iiky in range(len(iky_list)):
                my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[iiky]]))
                #plt.semilogy(t_collec[iiky], sum_phi2_collec[iiky], color=my_colorlist[iiky], linewidth=3.0)
                plt.plot(t, np.log(sum_phi2bloon[iiky][idmid]), color=my_colorlist[iiky], linewidth=3.0)
        plt.legend(my_legend)
        pdfname = 'floquet_sum_vs_t_all_ky' + '_dmid_' + str(my_dmid[idmid]) 
        pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
        plt.savefig(pdfname)
        plt.clf()
        plt.cla()

        plt.figure(figsize=(12,8))
        plt.xlabel('$t$')
        plt.ylabel('$\\max_{K_x}\\vert \\langle\\phi\\rangle_\\theta \\vert ^2$')
        plt.grid(True)
        my_legend = []
        my_colorlist = []
        # only use blue
        for iiky in range(len(iky_list)):
            my_colorlist.append(gplots.myblue)
        #my_colorlist = plt.cm.YlOrBr(np.linspace(0.2,1,len(iky_list))) # for newalgo
        #my_colorlist = plt.cm.YlGnBu(np.linspace(0.2,1,len(iky_list))) # for oldalgo
        if skip_init:
            for iiky in range(len(iky_list)):
                plt.semilogy(t_collec[iiky][idmid], max_phi2_collec[iiky][idmid], color=my_colorlist[iiky], linewidth=3.0)
                my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[iiky]]))
                # Add fits for average and maximum growthrates
                plt.semilogy(t_collec[iiky][idmid], np.exp(2.0*slope_max[iiky][idmid]*t_collec[iiky][idmid]+offset_max[iiky][idmid]),\
                        color=my_colorlist[iiky], linewidth=3.0, linestyle='--')
                if g_exb != 0.0:
                    my_legend.append('$\\langle\\gamma\\rangle_t = {:.3f}$'.format(slope_max[iiky][idmid]))
                    bot, top = plt.ylim()
                    plt.semilogy(t_collec[iiky][idmid], max_phi2bloon[iiky][idmid][it_gamma_max[iiky][idmid]]/max_phi2bloon[iiky][idmid][it_start] * \
                            np.exp(2.0*gamma_max[iiky][idmid]*(t_collec[iiky][idmid]-t[it_gamma_max[iiky][idmid]])),\
                            color=my_colorlist[iiky], linewidth=3.0, linestyle=':')
                    my_legend.append('$\\gamma_{max} '+'= {:.3f}$'.format(gamma_max[iiky][idmid]))
                    plt.ylim(bot,top)
                else:
                    my_legend.append('$\\gamma = {:.3f}$'.format(slope_max[iiky][idmid]))
        else:
            for iiky in range(len(iky_list)):
                plt.semilogy(t, max_phi2bloon[iiky][idmid], color=my_colorlist[iiky], linewidth=3.0)
                my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[iiky]]))
                # Add fits for average and maximum growthrates
                plt.semilogy(t, max_phi2bloon[iiky][idmid][it_start]*np.exp(2.0*slope_max[iiky][idmid]*t+offset_max[iiky][idmid]),\
                        color=my_colorlist[iiky], linewidth=3.0, linestyle='--')
                if g_exb != 0.0:
                    my_legend.append('$\\langle\\gamma\\rangle_t = {:.3f}$'.format(slope_max[iiky][idmid]))
                    bot, top = plt.ylim()
                    plt.semilogy(t, max_phi2bloon[iiky][idmid][it_gamma_max[iiky][idmid]]*np.exp(2.0*gamma_max[iiky][idmid]*(t-t[it_gamma_max[iiky][idmid]])),\
                            color=my_colorlist[iiky], linewidth=3.0, linestyle=':')
                    my_legend.append('$\\gamma_{max} '+'= {:.3f}$'.format(gamma_max[iiky][idmid]))
                    plt.ylim(bot,top)
                else:
                    my_legend.append('$\\gamma = {:.3f}$'.format(slope_max[iiky][idmid]))
        plt.legend(my_legend)
        pdfname = 'floquet_max_vs_t_all_ky' + '_dmid_' + str(my_dmid[idmid]) 
        pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
        plt.savefig(pdfname)
        plt.clf()
        plt.cla()

        for iiky in range(len(iky_list)):
        
            iky = iky_list[iiky]

            ## set up time stepping for snapshots and movies
            max_it_for_snap = nt
            it_step_for_snap = 10 # Adapt this
            max_it_for_mov = nt
            it_step_for_mov = 10

            ## Plot real frequency of connected chain, vs kxbar
            # Save snapshots
            tmp_pdf_id = 1
            pdflist = []
            for it in range(0,max_it_for_snap,it_step_for_snap):
                l1, = plt.plot(kxbarbloon[iiky][idmid][it],omegabloon[iiky][idmid][it], marker='o', color=gplots.myblue, \
                        markersize=12, markerfacecolor=gplots.myblue, markeredgecolor=gplots.myblue, linewidth=3.0)
                plt.xlabel('$\\rho\\bar{k}_x$')
                plt.ylabel('$\\omega$'+' '+'$[v_{thr}/r_r]$')
                plt.grid(True)
                ax = plt.gca()
                ax.set_title('$k_y={:.2f}$'.format(ky[iky]) + ', $t={:.2f}$'.format(t[it]))
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplots.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1

            merged_pdfname = 'omega_snaps_dmid_'+str(my_dmid[idmid])
            gplots.merge_pdfs(pdflist, merged_pdfname, run, ifile)
            plt.clf()
            plt.cla()

            if (phi_t_present):

                # find global min and max of ballooning angle
                bloonang_min = 0.
                bloonang_max = 0.
                for it in range(max_it_for_snap):
                    if np.min(bloonang[iiky][idmid][it]) < bloonang_min:
                        bloonang_min = np.min(bloonang[iiky][idmid][it])
                    if np.max(bloonang[iiky][idmid][it]) > bloonang_max:
                        bloonang_max = np.max(bloonang[iiky][idmid][it])

                ## Save snapshots
                tmp_pdf_id = 1
                pdflist = []
                for it in range(0,max_it_for_snap,it_step_for_snap):
                    l1, = plt.plot(bloonang[iiky][idmid][it],phi2bloon[iiky][idmid][it], marker='o', color=gplots.myblue, \
                            markersize=12, markerfacecolor=gplots.myblue, markeredgecolor=gplots.myblue, linewidth=3.0)
                    l2, = plt.plot(bloonang_bndry[iiky][idmid][it],phi2bloon_discont[iiky][idmid][it], linestyle='', \
                            marker='o', markersize=8, markerfacecolor='r', markeredgecolor='r')
                    plt.xlabel('$\\theta -\\theta_0^*$') # NDCPARAM: check for plot_against_theta0_star
                    plt.grid(True)
                    plt.gca().set_xlim(bloonang_min,bloonang_max)
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
                    ymin = np.amin(phi2bloon[iiky][idmid][it])
                    ymax = np.amax(phi2bloon[iiky][idmid][it])
                    ax = plt.gca()
                    ax.set_ylim(ymin,ymax)
                    ax.set_title('$k_y={:.2f}, t={:.2f}$'.format(ky[iky],t[it]))
                    ax.legend(['$\\vert \\phi \\vert ^2$', '$\\Delta\\vert \\phi \\vert ^2$'])
                    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                    gplots.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdfname)
                    tmp_pdf_id = tmp_pdf_id+1

                merged_pdfname = 'phibloon_snaps_dmid_'+str(my_dmid[idmid])
                gplots.merge_pdfs(pdflist, merged_pdfname, run, ifile)
                plt.clf()
                plt.cla()

                if (make_movies):

                    # find global min and max of ballooning angle
                    bloonang_min = 0.
                    bloonang_max = 0.
                    for it in range(max_it_for_mov):
                        if np.min(bloonang[iiky][idmid][it]) < bloonang_min:
                            bloonang_min = np.min(bloonang[iiky][idmid][it])
                        if np.max(bloonang[iiky][idmid][it]) > bloonang_max:
                            bloonang_max = np.max(bloonang[iiky][idmid][it])
                    
                    ## movie of phi2 vs ballooning angle over time
                    moviename = 'phi_bloon' + '_iky_' + str(iky) + '_dmid_' + str(my_dmid[idmid])
                    moviename = run.out_dir + moviename + '_' + run.fnames[ifile] + '.mp4'
                   
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
                        #sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1))) # comment out on HPC
                        xdata1 = bloonang[iiky][idmid][it]
                        ydata1 = phi2bloon[iiky][idmid][it]
                        l1.set_data(xdata1,ydata1)
                        ymin = np.amin(phi2bloon[iiky][idmid][it])
                        ymax = np.amax(phi2bloon[iiky][idmid][it])
                        ax = plt.gca()
                        ax.set_ylim(ymin,ymax)
                        ax.set_title('$k_y={:.2f}, t={:.2f}$'.format(ky[iky],t[it]))
                        # Update discontinuities at 2pi interfaces
                        xdata2 = bloonang_bndry[iiky][idmid][it]
                        ydata2 = phi2bloon_discont[iiky][idmid][it]
                        l2.set_data(xdata2,ydata2)
                        return l1, l2

                    mov = anim.FuncAnimation(myfig,update_mov,init_func=init_mov, \
                            frames=range(0,max_it_for_mov,it_step_for_mov),blit=False)
                    writer = anim.writers['ffmpeg'](fps=10,bitrate=-1,codec='libx264')
                    mov.save(moviename,writer=writer,dpi=100)
                    plt.clf()
                    plt.cla()

                    ## movie of phi2 jump at interfaces between 2pi domains
                    moviename = 'phijump' + '_iky_' + str(iky) + '_dmid_' + str(my_dmid[idmid])
                    moviename = run.out_dir + moviename + '_' + run.fnames[ifile] + '.mp4'

                    print("\ncreating movie of phijump vs ballooning angle ...")
                    xdata1, ydata1 = [], []
                    l1, = plt.plot([],[], marker='o', color=gplots.myred, \
                            markersize=12, markerfacecolor=gplots.myred, markeredgecolor=gplots.myred, linewidth=3.0)
                    plt.xlabel('$\\theta -\\theta_0^*$') # NDCPARAM: check for plot_against_theta0_star
                    plt.ylabel('$\\Delta\\vert \\phi \\vert ^2/\\vert \\phi \\vert ^2$')
                    plt.grid(True)
                    plt.gca().set_xlim(bloonang_min,bloonang_max)
                    plt.gca().set_ylim(0,1)
                    # Initialize lines
                    def init_mov_jump():
                        l1.set_data([], [])
                        return l1
                    # Update lines
                    def update_mov_jump(it):
                        #sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1))) # comment out for HPC
                        # Update discontinuities at 2pi interfaces
                        xdata1 = bloonang_bndry[iiky][idmid][it]
                        ydata1 = phi2bloon_jump[iiky][idmid][it]
                        plt.gca().set_title('$k_y={:.2f}, t={:.2f}$'.format(ky[iky],t[it]))
                        l1.set_data(xdata1,ydata1)
                        return l1

                    mov = anim.FuncAnimation(myfig,update_mov_jump,init_func=init_mov_jump, \
                            frames=range(0,max_it_for_mov,it_step_for_mov),blit=False)
                    writer = anim.writers['ffmpeg'](fps=10,bitrate=-1,codec='libx264')
                    mov.save(moviename,writer=writer,dpi=100)
                    plt.clf()
                    plt.cla()

                    ## Save snapshots
                    #for it in range(0,max_it_for_mov,it_step_for_mov):
                    #    l1, = plt.plot(bloonang_bndry[iiky][it],phi2bloon_jump[iiky][it], marker='o', color=gplots.myblue, \
                    #            markersize=12, markerfacecolor=gplots.myblue, markeredgecolor=gplots.myblue, linewidth=3.0)
                    #    plt.xlabel('$\\theta -\\theta_0^*$') # NDCPARAM: check for plot_against_theta0_star
                    #    plt.ylabel('$\\Delta\\vert \\phi \\vert ^2/\\vert \\phi \\vert ^2$')
                    #    plt.grid(True)
                    #    plt.gca().set_xlim(bloonang_min,bloonang_max)
                    #    ax = plt.gca()
                    #    ax.set_ylim(0,1)
                    #    ax.set_title('$k_y={:.2f}, t={:.2f}$'.format(ky[iky],t[it]))
                    #    plt.savefig('phidiscont_snap_{:,d}.pdf'.format(it))
                    #    plt.clf()
                    #    plt.cla()

                    print("\n... movies completed.")


    for iiky in range(len(iky_list)):
    
        iky = iky_list[iiky]

        # plot phi2 vs t for each kx
        plt.title('$k_y={:.2f}$'.format(ky[iky]))
        plt.xlabel('$t\\ [r_r/v_{thr}]$')
        my_ylabel = '$\\ln \\left(\\vert \\langle \\phi \\rangle_\\theta \\vert ^2\\right)$'
        plt.ylabel(my_ylabel)
        plt.grid(True)
        my_colorlist = plt.cm.plasma(np.linspace(0,1,kx_bar.size))
        my_legend = []
        kxs_to_plot=kx_bar
        for ikx in range(kx_bar.size):
            if kx_bar[ikx] in kxs_to_plot:
                plt.plot(t, np.log(phi2[:,iky,ikx]), color=my_colorlist[ikx])
                #my_legend.append('$\\rho_i\\bar{k}_x = '+str(kx_bar[ikx])+'$')
        #plt.legend(my_legend)
        axes=plt.gca()

        pdfname = 'phi2_by_kx_iky_{:d}'.format(iky)
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
    task_space.dmid = my_vars['dmid']
    task_space.ky = my_vars['ky']
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
    # Here, assume that all files have the same iky_list & ky
    iky_list = full_space[0]['floquet'].iky_list
    dmid = full_space[0]['floquet'].dmid
    ky = full_space[0]['floquet'].ky
    
    sum_phi2 = []
    t = []
    delt = np.zeros(len(run.fnames))
    dkx = np.zeros(len(run.fnames))
    slope = np.zeros(len(run.fnames))
    
    if not run.no_plot:

        ifile_dummy=0

        for idmid in range(len(dmid)):

            for iiky in range(len(iky_list)):

                iky = iky_list[iiky]

                for ifile in range(len(run.fnames)):
                    
                    Tf = full_space[ifile]['floquet'].Tf
                    delt[ifile] = full_space[ifile]['floquet'].delt
                    dkx[ifile] = full_space[ifile]['floquet'].dkx
                    nwrite = full_space[ifile]['floquet'].nwrite

                    it_start = int(round((N_start*Tf/delt[ifile])/nwrite))

                    sum_phi2_tmp = np.zeros(len(full_space[ifile]['floquet'].sum_phi2bloon[iiky][idmid])-it_start)
                    for it in range(sum_phi2_tmp.size):
                        sum_phi2_tmp[it] = full_space[ifile]['floquet'].sum_phi2bloon[iiky][idmid][it_start+it]
                    sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
                    sum_phi2.append(sum_phi2_tmp)
                    
                    t_tmp = np.zeros(len(full_space[ifile]['floquet'].t)-it_start)
                    for it in range(t_tmp.size):
                        t_tmp[it] = full_space[ifile]['floquet'].t[it_start+it]
                    t.append(t_tmp)

                    [a,dummy] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
                    slope[ifile] = a
    
                print('Slopes for ky={:.3f}:'.format(ky[iky]))
                print(slope)
                plt.figure(figsize=(12,8))
                
                # Plot phi2 summed along chain
                plt.xlabel('$t [a/v_{thi,i}]$')
                plt.ylabel('$\\ln \\left(\\sum_{k_x}\\vert \\langle\\hat{\\varphi}\\rangle_\\theta \\vert ^2\\right)$')
                if len(iky_list)>1:
                    plt.title('$\\rho k_y = {:.3f}$'.format(ky[iky]))
                plt.grid(True)
                my_legend = []
                #my_colorlist = plt.cm.YlOrBr(np.linspace(0.2,1,len(run.fnames))) # for newalgo
                my_colorlist = plt.cm.YlGnBu(np.linspace(0.2,1,len(run.fnames))) # for oldalgo
                for ifile in range(len(run.fnames)):
                    #my_legend.append('$\\Delta t =$'+str(full_space[ifile]['floquet'].delt))
                    my_legend.append('$\\rho(\\Delta k_x) = {:.3f}$'.format(full_space[ifile]['floquet'].dkx))
                    plt.plot(t[ifile], np.log(sum_phi2[ifile]), color=my_colorlist[ifile], linewidth=3.0)
                plt.legend(my_legend)
                axes = plt.gca()
                axes.set_ylim([-0.5, 13.75])

                # Plot growthrates within phi2 plot
                subaxes = plt.axes([0.65, 0.25, 0.3, 0.25])
                subaxes.tick_params(labelsize=18)
                plt.xlabel('$\\rho(\\Delta k_x)$',fontsize=20)
                #plt.ylabel('$\\langle \\gamma \\rangle_t$',fontsize=20)
                plt.title('Time averaged growth-rate',fontsize=20)
                plt.grid(True)
                plt.plot(dkx, slope, marker='o', color='black', \
                        markersize=10, markerfacecolor='none', markeredgecolor='black', linewidth=2.0)
                gamma_converged = 0.06418364
                plt.axhline(y=gamma_converged, color='grey', linestyle='--', linewidth=2.0) # for oldalgo
                subaxes.set_ylim([0.06, 0.13])
                
                plot_name = run.scan_name
                if len(iky_list)>1:
                    plot_name = plot_name + '_ky_{:.3f}'.format(ky[iky])
                plot_name = plot_name + '_dmid_' + str(dmid[idmid])
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
