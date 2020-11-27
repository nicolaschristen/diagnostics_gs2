from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as anim
import numpy as np
from math import pi
from math import ceil
import imageio
import os
import sys
import pickle
from PyPDF2 import PdfFileMerger, PdfFileReader
import scipy.interpolate as scinterp
import scipy.optimize as opt

import gs2_plotting as gplot

# A twist and shift chain is identified by dmid = 0,
# which is the number of dkx between kx=0 and the
# smallest kx>0 that is a member of the chain, at t=0.

ndec_tt0 = 2

def my_task_single(ifile, run, myin, myout, mytime, task_space):  





    # vvv User parameters vvv
    
    # select chains
    dmid_list = []

    # Select t/tfinal for plot of phi vs ballooning angle
    tRatio_toPlot = [1.0]

    # make snapshots/movies along ballooning angle ?
    make_plots = True

    plot_phi_vs_t = True
    plot_phi_vs_tt0 = True
    plot_phi_vs_tt0_log = False

    make_snaps = False
    itSnap_min = 0
    itSnap_max = -1
    itSnap_step = 10

    make_movies = False
    itMov_min = 0
    itMov_max = -1
    itMov_step = 1

    make_movie_discont = False

    # ^^^ User parameters ^^^





    t = myout['t']
    delt = myin['knobs']['delt']
    nt = t.size
    nwrite = myin['gs2_diagnostics_knobs']['nwrite']

    theta = myout['theta']
    ntheta = theta.size
    
    try:
        g_exbfac = myin['dist_fn_knobs']['g_exbfac']
    except:
        g_exbfac = 1.0
    g_exb = g_exbfac * myin['dist_fn_knobs']['g_exb']

    shat = myin['theta_grid_parameters']['shat']
    jtwist = int(myin['kt_grids_box_parameters']['jtwist'])

    kx_gs2 = myout['kx']
    # We only consider the first non-zero ky
    ky = myout['ky'][1]
    dky = 1./myin['kt_grids_box_parameters']['y0']
    dkx = 2.*pi*abs(shat)*dky/jtwist
    nakx = kx_gs2.size
    ikx_max = int(round((nakx-1)/2))
    ikx_min = ikx_max+1
    # Index of kx=0
    ikx0 = (nakx-1)//2
    
    # Number of t-steps before ExB re-map
    if g_exb != 0.0:
        N_per_remap = max(1, abs(int(round(dkx/(g_exb*delt*dky)))))
    else:
        N_per_remap = np.nan
    # Floquet period
    if g_exb != 0.0:
        Tf = abs(2*pi*shat/g_exb)
    else:
        Tf = np.nan
    # Number of t-steps in Floquet period
    if g_exb != 0.0:
        Nf = int(round(Tf/(delt*nwrite)))
    else:
        Nf = np.nan

    # Read data from output
    phi2_gs2 = np.squeeze(myout['phi2_by_mode'][:,1,:])
    omega_gs2 = np.squeeze(myout['omega_average'][:,1,:,0]) # real frequency
    phi_t_present = myout['phi_t_present']
    if phi_t_present:
        phi2_bytheta_gs2 = np.sum(np.power(myout['phi_t'],2), axis=4)
        phi2_bytheta_gs2 = np.squeeze(phi2_bytheta_gs2[:,1,:,:])
    try:
        Qe = myout['es_heat_flux'][:,1]
        Qi = myout['es_heat_flux'][:,0]
        Qratio = Qe/Qi
        Qratio_avg = mytime.timeavg(Qratio)
    except:
        Qratio_avg = 'NaN'
   
    # Sorting kx indices to become monotonic
    kx_bar = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))
    phi2 = np.concatenate((phi2_gs2[:,ikx_min:], phi2_gs2[:,:ikx_min]), axis=1)
    omega = np.concatenate((omega_gs2[:,ikx_min:], omega_gs2[:,:ikx_min]), axis=1)
    if phi_t_present:
        phi2_bytheta = np.concatenate((phi2_bytheta_gs2[:,ikx_min:,:], phi2_bytheta_gs2[:,:ikx_min,:]), axis=1)

    # Check if we want to process all -pi < theta0 < pi (ie if dmid_list=[])
    if not dmid_list:
        dmid_list = [0]
        ikx = 1
        while abs(ikx*dkx/(shat*ky)) <= pi:
            dmid_list.insert(0,-ikx)
            dmid_list.append(ikx)
            ikx += 1

    # Check if the chain has at least one connection.
    # If not, we will follow the right-most (g_exb>0)
    # or left-most (g_exb<0) chain until it falls off the grid.
    if jtwist > nakx:
        if g_exb != 0:
            dmid_list = [int(np.sign(g_exb)*(nakx-1)/2)]
            nt = int((nakx-1) * np.floor(N_per_remap/nwrite))
            if nt == 0:
                print('\n============== WARNING ==============\n'+ \
                        'This chain has no connections, and is\n'+ \
                        '   dropped from the simulation at    \n'+ \
                        '         t < nwrite*delt             \n'+ \
                        '=====================================\n' )

    # Get kx and kx_star from kx_bar
    kx = np.zeros((nt,nakx))
    kx_star = np.zeros((nt,nakx))
    # @ it = 0, kx = kx_bar
    kx[0,:] = kx_bar
    kx_star[0,:] = kx_bar
    # First step is taken with 0.5*dt
    # Other steps taken with full dt
    for it in range(1,nt):
        ikx_shift = int(round(g_exb*ky*delt*(nwrite*it-0.5)/dkx))
        for ikx in range(nakx):
            kx[it,ikx] = kx_bar[ikx] + ikx_shift*dkx
            kx_star[it,ikx] = kx[it,ikx] - g_exb*ky*delt*(nwrite*it-0.5)

    # Construct theta0 grids
    itheta0_list = [dmid + ikx0 for dmid in dmid_list]
    theta0 = kx_bar/(shat*ky)
    theta0_star = np.zeros((nt,nakx))
    theta0_star[0,:] = theta0
    for it in range(1,nt):
        for ikx in range(nakx):
            theta0_star[it,ikx] = (kx_bar[ikx]-g_exb*ky*delt*(nwrite*it-0.5))/(shat*ky)
    
    # Lists containing all ballooning chains at every time step
    ikx_members = []
    ikx_prevmembers = []
    bloonang = []
    bloonang_bndry = []
    phi2bloon = []
    phi2bloon_discont = []
    phi2bloon_jump = []
    max_phi2bloon = []
    sum_phi2bloon = []
    omegabloon = []
    gamma = []
    kx_star_for_gamma = []

    for dmid in dmid_list:

        # Lists containing all time steps for the current chain
        ikx_members_chain = []
        ikx_prevmembers_chain = []
        bloonang_chain = []
        bloonang_bndry_chain = []
        phi2bloon_chain = []
        phi2bloon_discont_chain = []
        phi2bloon_jump_chain = []
        max_phi2bloon_chain = []
        sum_phi2bloon_chain = []
        gamma_chain = []
        omegabloon_chain = []

        for it in range(nt):

            # Lists containing the current time step of the current chain
            phi2bloon_now = []
            phi2bloon_discont_now = []
            phi2bloon_jump_now = []
            omegabloon_now = []

            # Determine (or, if gexb/=0, update) the 
            # collection of ikx's that are included
            # in the chain at time step it (ikx_members_now)
            # and at time step it-1 (ikx_prevmembers_now).

            if it==0 or g_exb != 0.0:

                # Lists containing the current time step of the current chain
                ikx_members_now = []
                ikx_prevmembers_now = []
                bloonang_now = []
                bloonang_bndry_now = []

                if it==0:
                    ikx_shift = 0
                    ikx_shift_old = 0
                elif it==1:
                    ikx_shift = int(round(g_exb*ky*delt*(nwrite*it-0.5)/dkx))
                    ikx_shift_old = 0
                else:
                    ikx_shift = int(round(g_exb*ky*delt*(nwrite*it-0.5)/dkx))
                    ikx_shift_old = int(round(g_exb*ky*delt*(nwrite*(it-1)-0.5)/dkx))

                # ikx such that our chain includes kxstar(t=0) = dmid*dkx
                ikx = ikx0 - ikx_shift + dmid
                ikxprev = ikx0 - ikx_shift_old + dmid

                while (ikx >= nakx): # moving from the right to first connected kx within the set in GS2
                    ikx = ikx-jtwist
                    ikxprev = ikxprev-jtwist
                while (ikx >= 0):
                    ikx_members_now.append(ikx)
                    if ikxprev >= nakx:
                        ikx_prevmembers_now.append(np.nan)
                    elif ikxprev < 0:
                        ikx_prevmembers_now.append(np.nan)
                    else:
                        ikx_prevmembers_now.append(ikxprev)
                    ikx = ikx-jtwist
                    ikxprev = ikxprev-jtwist

                ikx = ikx0 - ikx_shift + dmid + jtwist
                ikxprev = ikx0 - ikx_shift_old + dmid + jtwist
                while (ikx < 0): # moving from the left to first connected kx within the set in GS2
                    ikx = ikx+jtwist
                    ikxprev = ikxprev+jtwist
                while (ikx < nakx):
                    ikx_members_now.append(ikx)
                    if ikxprev >= nakx:
                        ikx_prevmembers_now.append(np.nan)
                    elif ikxprev < 0:
                        ikx_prevmembers_now.append(np.nan)
                    else:
                        ikx_prevmembers_now.append(ikxprev)
                    ikx = ikx+jtwist
                    ikxprev = ikxprev+jtwist

                # sort ikx of chain members at time it in left-to-right order (shat>0: descending, shat<0: ascending)
                # sort time it-1 accordingly
                idx_sort = sorted(range(len(ikx_members_now)), key=lambda k: ikx_members_now[k],reverse=(shat>0.))
                ikx_members_now = [ikx_members_now[idx] for idx in idx_sort]
                ikx_prevmembers_now = [ikx_prevmembers_now[idx] for idx in idx_sort]

                member_range = range(len(ikx_members_now))
                
                # compute ballooning angle
                for imember in member_range:
                    if imember < len(ikx_members_now)-1:
                        b_ang_bndry = pi-kx_star[it,ikx_members_now[imember]]/(shat*ky)
                        bloonang_bndry_now.append(b_ang_bndry)
                    for itheta in range(ntheta):
                        b_ang = theta[itheta] - kx_star[it,ikx_members_now[imember]]/(shat*ky)
                        bloonang_now.append(b_ang)

            if phi_t_present:
                
                for imember in member_range:
                    
                    # construct chain of phi2
                    for itheta in range(ntheta):
                        phi2bloon_now.append(phi2_bytheta[it,ikx_members_now[imember],itheta])
                    
                    # construct chain of real frequency
                    omegabloon_now.append(omega[it,ikx_members_now[imember]])

                    # Saving discontinuities
                    if imember < len(ikx_members_now)-1:
                        phi2_l = phi2_bytheta[it,ikx_members_now[imember],-1]
                        phi2_r = phi2_bytheta[it,ikx_members_now[imember+1],0]
                        discont = abs(phi2_r-phi2_l)
                        phi2bloon_discont_now.append(discont)
                        jump = abs((phi2_r-phi2_l)/max(phi2_l,phi2_r))
                        phi2bloon_jump_now.append(jump)

            # Add this time step to the current chain
            ikx_members_chain.append(ikx_members_now)
            ikx_prevmembers_chain.append(ikx_prevmembers_now)
            bloonang_chain.append(bloonang_now)
            bloonang_bndry_chain.append(bloonang_bndry_now)
            phi2bloon_chain.append(phi2bloon_now)
            phi2bloon_discont_chain.append(phi2bloon_discont_now)
            phi2bloon_jump_chain.append(phi2bloon_jump_now)
            omegabloon_chain.append(omegabloon_now)
            max_phi2bloon_chain.append(max(phi2bloon_now))
            sum_phi2bloon_chain.append(sum(phi2bloon_now))

        # Add this chain to the full collection
        ikx_members.append(ikx_members_chain)
        ikx_prevmembers.append(ikx_prevmembers_chain)
        bloonang.append(bloonang_chain)
        bloonang_bndry.append(bloonang_bndry_chain)
        phi2bloon.append(phi2bloon_chain)
        phi2bloon_discont.append(phi2bloon_discont_chain)
        phi2bloon_jump.append(phi2bloon_jump_chain)
        max_phi2bloon.append(max_phi2bloon_chain)
        sum_phi2bloon.append(sum_phi2bloon_chain)
        omegabloon.append(omegabloon_chain)

    # Start comparing simulations at time-step it_start

    if g_exb != 0.0:

        # Number of full Floquet periods in simulation (first might be incomplete)
        nTf = int((nt-1)//Nf)
        # Add the incomplete interval, if it has more than one time-step
        if (nt-1)%Nf > 1:
            nTf += 1
        # Indices of start/end time corresponding to each interval
        it_Tfend = [(nt-1)-(nTf-1-i)*Nf for i in range(nTf)]
        it_Tfstart = [max(1,i-Nf) for i in it_Tfend]
        # Number of time steps in every interval
        nt_Tf = [it_Tfend[i]-it_Tfstart[i] for i in range(nTf)]

        # Decide how many intervals to skip at start of simulation
        if nTf==0:
            skip_nTf = 0
        elif nt_Tf[0] < 0.8*Nf and nTf>=2:
            skip_nTf = 2
        else:
            skip_nTf = 1

        it_start = it_Tfstart[skip_nTf]

    else:

        fac = 0.5
        it_start = round(fac*nt)

    t_collec = []
    max_phi2_collec = []
    sum_phi2_collec = []
    
    # Compute <gamma>_t
    gamma_avg = np.zeros(len(dmid_list))
    offset_avg = np.zeros(len(dmid_list))
    gamma_avg_fromSum = np.zeros(len(dmid_list))
    offset_avg_fromSum = np.zeros(len(dmid_list))

    for idmid in range(len(dmid_list)):

        # Selected time window
        t_tmp = np.zeros(nt-it_start)
        for it in range(t_tmp.size):
            t_tmp[it] = t[it_start+it]
        t_collec.append(t_tmp)

        # Gamma from max(phi2)
        max_phi2_tmp = np.zeros(len(max_phi2bloon[idmid])-it_start)
        for it in range(max_phi2_tmp.size):
            max_phi2_tmp[it] = max_phi2bloon[idmid][it_start+it]
        if it_start > 0:
            max_phi2_tmp = max_phi2_tmp/max_phi2_tmp[0]
        max_phi2_collec.append(max_phi2_tmp)

        [gam,offset] = leastsq_lin(t_tmp,np.log(max_phi2_tmp))
        gamma_avg[idmid] = gam/2. # divide by 2 because fitted square
        offset_avg[idmid] = offset

        # Gamma from sum(phi2)
        sum_phi2_tmp = np.zeros(len(sum_phi2bloon[idmid])-it_start)
        for it in range(sum_phi2_tmp.size):
            sum_phi2_tmp[it] = sum_phi2bloon[idmid][it_start+it]
        if it_start > 0:
            sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
        sum_phi2_collec.append(sum_phi2_tmp)

        [gam,offset] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
        gamma_avg_fromSum[idmid] = gam/2. # divide by 2 because fitted square
        offset_avg_fromSum[idmid] = offset

    # At this point:
    # fit_avg(t) ~ phi2(tstart) * exp[2*gam*t + offset]



    # Compute instantaneous and maximum growthrates from ln(phi2_max)
    # For each chain and at every time, gamma_inst will
    # have a corresponding theta0* stored in theta0_star_for_inst

    # If gexb = 0, growthrate is cst in time
    # so the following variables are unused and should not be considered

    if g_exb != 0.0:

        it_gamma_max = np.zeros(len(dmid_list))
        it_gamma_max_fromSum = np.zeros(len(dmid_list))
        gamma_max = np.zeros(len(dmid_list))
        gamma_max_fromSum = np.zeros(len(dmid_list))
        gamma_inst = []
        gamma_inst_fromSum = []
        theta0_star_for_inst = []

        for idmid in range(len(dmid_list)):

            gamma_max_by_Tf = np.zeros(nTf)
            gamma_max_fromSum_by_Tf = np.zeros(nTf)
            it_gamma_max_by_Tf = np.zeros(nTf)
            it_gamma_max_fromSum_by_Tf = np.zeros(nTf)

            # Pick refined, evenly spaced theta0 grid to apply nearest neighbour
            # interpolation to gamma_inst for each Tf interval.
            ntt0_fine = 1001
            tt0_fine = np.linspace(-pi,pi,ntt0_fine)
            gamma_inst_by_dmid = [0]*ntt0_fine
            gamma_inst_fromSum_by_dmid = [0]*ntt0_fine

            for iTf in range(skip_nTf,nTf):

                gamma_inst_by_dmid_Tf = []
                gamma_inst_fromSum_by_dmid_Tf = []
                theta0_star_for_inst_by_dmid_Tf = []

                for it in range(it_Tfstart[iTf],it_Tfend[iTf]):

                    # Factor of 0.5 because we fit phi^2
                    gamma_max_tmp = 0.5 * 1./(2*delt*nwrite) * \
                            ( np.log(max_phi2bloon[idmid][it+1]) - np.log(max_phi2bloon[idmid][it-1]) )
                    gamma_max_fromSum_tmp = 0.5 * 1./(2*delt*nwrite) * \
                            ( np.log(sum_phi2bloon[idmid][it+1]) - np.log(sum_phi2bloon[idmid][it-1]) )

                    # Fill instantaneous growthrate and corresponding theta0_star
                    gamma_inst_by_dmid_Tf.append(gamma_max_tmp)
                    gamma_inst_fromSum_by_dmid_Tf.append(gamma_max_fromSum_tmp)

                    # Update maximum growthrate if needed
                    # First from max(phi2)
                    if (gamma_max_tmp > gamma_max_by_Tf[iTf]):
                        it_gamma_max_by_Tf[iTf] = it
                        gamma_max_by_Tf[iTf] = gamma_max_tmp
                    # Then from sum(phi2)
                    if (gamma_max_fromSum_tmp > gamma_max_fromSum_by_Tf[iTf]):
                        it_gamma_max_fromSum_by_Tf[iTf] = it
                        gamma_max_fromSum_by_Tf[iTf] = gamma_max_fromSum_tmp

                    # Determine current theta0_star associated with this chain
                    tt0_tmp = theta0_star[it,itheta0_list[idmid]]
                    # Shift it to [-pi,+pi]
                    n = int(round(tt0_tmp/(2.0*pi)))
                    tt0_tmp -= 2*pi*n
                    theta0_star_for_inst_by_dmid_Tf.append(tt0_tmp)

                # Get theta0_star in ascending order
                idx_sort = np.argsort(theta0_star_for_inst_by_dmid_Tf)
                theta0_star_for_inst_by_dmid_Tf = [theta0_star_for_inst_by_dmid_Tf[idx] for idx in idx_sort]
                # Sort gamma_inst accordingly
                gamma_inst_by_dmid_Tf = [gamma_inst_by_dmid_Tf[idx] for idx in idx_sort]
                gamma_inst_fromSum_by_dmid_Tf = [gamma_inst_fromSum_by_dmid_Tf[idx] for idx in idx_sort]

                # Average gamma_inst over all Tf intervals,
                # except the ones we decided to skip
                gamma_inst_by_dmid_Tf_refined = gplot.nearNeighb_interp_1d(theta0_star_for_inst_by_dmid_Tf, gamma_inst_by_dmid_Tf, tt0_fine)
                gamma_inst_fromSum_by_dmid_Tf_refined = gplot.nearNeighb_interp_1d(theta0_star_for_inst_by_dmid_Tf, gamma_inst_fromSum_by_dmid_Tf, tt0_fine)
                for i in range(ntt0_fine):
                    gamma_inst_by_dmid[i] += gamma_inst_by_dmid_Tf_refined[i]/(nTf-skip_nTf)
                    gamma_inst_fromSum_by_dmid[i] += gamma_inst_fromSum_by_dmid_Tf_refined[i]/(nTf-skip_nTf)

            # Append theta0_star and gamma_inst to full lists
            theta0_star_for_inst.append(tt0_fine)
            gamma_inst.append(gamma_inst_by_dmid)
            gamma_inst_fromSum.append(gamma_inst_fromSum_by_dmid)

            # Average max gamma over all Tf intervals,
            # except the ones we decided to skip
            for iTf in range(skip_nTf,nTf):
                gamma_max[idmid] += gamma_max_by_Tf[iTf]
                gamma_max_fromSum[idmid] += gamma_max_fromSum_by_Tf[iTf]
            gamma_max[idmid] = gamma_max[idmid]/(nTf-skip_nTf)
            gamma_max_fromSum[idmid] = gamma_max_fromSum[idmid]/(nTf-skip_nTf)

            # Pick it_gamma_max at the end of the simulation (used for plotting only)
            it_gamma_max[idmid] = it_gamma_max_by_Tf[-1]
            it_gamma_max_fromSum[idmid] = it_gamma_max_fromSum_by_Tf[-1]

        it_gamma_max = it_gamma_max.astype(int)
        it_gamma_max_fromSum = it_gamma_max_fromSum.astype(int)
        # At this point:
        # fit_max(t) ~ phi2(tstart) * exp[2*gamma_max*(t-t_gamma_max)]

    else:

        gamma_inst = []
        gamma_inst_fromSum = []
        theta0_star_for_inst = []
        for idmid in range(len(dmid_list)):
            gamma_inst.append([])
            gamma_inst_fromSum.append([])
            theta0_star_for_inst.append([])
        gamma_max = gamma_avg
        gamma_max_fromSum = gamma_avg_fromSum



    
    # Save quantities to file for scan plots

    fname = run.out_dir + run.fnames[ifile] + '.linbox.dat'

    with open(fname, 'wb') as outfile:

        vardict = {}

        vardict['ky'] = ky
        vardict['g_exb'] = g_exb
        vardict['Qratio_avg'] = Qratio_avg
        vardict['gamma_max'] = gamma_max
        vardict['gamma_max_fromSum'] = gamma_max_fromSum
        vardict['gamma_avg'] = gamma_avg
        vardict['gamma_avg_fromSum'] = gamma_avg_fromSum
        vardict['gamma_inst'] = gamma_inst
        vardict['gamma_inst_fromSum'] = gamma_inst_fromSum
        vardict['dmid_list'] = dmid_list
        vardict['itheta0_list'] = itheta0_list
        vardict['theta0'] = theta0
        vardict['theta0_star_for_inst'] = theta0_star_for_inst

        pickle.dump(vardict, outfile)





    
    # Plotting


    if make_plots:

        if plot_phi_vs_t:

            # Plot max(phi2) in chain vs time, one dmid per plot

            plt.figure(figsize=(12,8))

            tmp_pdf_id = 1
            pdflist = []
            for idmid in range(len(dmid_list)):
                my_legend = []
                plt.semilogy(t[0:nt], max_phi2bloon[idmid], color=gplot.myblue, linewidth=3.0)
                plt.title('$k_y=$'+gplot.str_ky(ky) + ', $\\theta_0=$'+gplot.str_tt0(theta0[itheta0_list[idmid]]))
                my_legend.append('$\\max_{K_x}\\vert \\langle\\varphi\\rangle_\\theta \\vert ^2$')
                # Add fits for average and maximum growthrates
                plt.semilogy(t[0:nt], max_phi2bloon[idmid][it_start]*np.exp(2.0*gamma_avg[idmid]*t[0:nt]+offset_avg[idmid]),\
                        color=gplot.myblue, linewidth=3.0, linestyle='--')
                if g_exb != 0.0:
                    my_legend.append('$\\langle\\gamma\\rangle_t = {:.3f}$'.format(gamma_avg[idmid]))
                    bot, top = plt.ylim()
                    plt.semilogy(t[0:nt], max_phi2bloon[idmid][it_gamma_max[idmid]]*np.exp(2.0*gamma_max[idmid]*(t[0:nt]-t[it_gamma_max[idmid]])),\
                            color=gplot.myblue, linewidth=3.0, linestyle=':')
                    my_legend.append('$\\gamma_{max} '+'= {:.3f}$'.format(gamma_max[idmid]))
                    plt.ylim(bot,top)
                else:
                    my_legend.append('$\\gamma = {:.3f}$'.format(gamma_avg[idmid]))
                plt.xlabel('$t$')
                plt.grid(True)
                gplot.legend_matlab(my_legend)
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplot.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1

            merged_pdfname = 'maxphi_vs_t'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
            plt.clf()
            plt.cla()



            # Plot sum(phi2) in chain vs time, one dmid per plot

            plt.figure(figsize=(12,8))

            tmp_pdf_id = 1
            pdflist = []
            for idmid in range(len(dmid_list)):
                my_legend = []
                plt.semilogy(t[0:nt], sum_phi2bloon[idmid], color=gplot.myblue, linewidth=3.0)
                plt.title('$k_y=$'+gplot.str_ky(ky) + ', $\\theta_0=$'+gplot.str_tt0(theta0[itheta0_list[idmid]]))
                my_legend.append('$\\sum_{K_x}\\vert \\langle\\varphi\\rangle_\\theta \\vert ^2$')
                # Add fits for average and maximum growthrates
                plt.semilogy(t[0:nt], sum_phi2bloon[idmid][it_start]*np.exp(2.0*gamma_avg_fromSum[idmid]*t[0:nt]+offset_avg_fromSum[idmid]),\
                        color=gplot.myblue, linewidth=3.0, linestyle='--')
                if g_exb != 0.0:
                    my_legend.append('$\\langle\\gamma\\rangle_t = {:.3f}$'.format(gamma_avg_fromSum[idmid]))
                    bot, top = plt.ylim()
                    plt.semilogy(t[0:nt],sum_phi2bloon[idmid][it_gamma_max_fromSum[idmid]]*np.exp(2.0*gamma_max_fromSum[idmid]*(t[0:nt]-t[it_gamma_max_fromSum[idmid]])),\
                            color=gplot.myblue, linewidth=3.0, linestyle=':')
                    my_legend.append('$\\gamma_{max} '+'= {:.3f}$'.format(gamma_max_fromSum[idmid]))
                    plt.ylim(bot,top)
                else:
                    my_legend.append('$\\gamma = {:.3f}$'.format(gamma_avg_fromSum[idmid]))
                plt.xlabel('$t$')
                plt.grid(True)
                gplot.legend_matlab(my_legend)
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplot.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id+1

            merged_pdfname = 'sumphi_vs_t'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
            plt.clf()
            plt.cla()



        # plot phi2 vs t for each kx

        plt.figure(figsize=(12,8))
        plt.title('$k_y={:.2f}$'.format(ky))
        plt.xlabel('$t\\ [r_r/v_{thr}]$')
        my_ylabel = '$\\ln \\left(\\vert \\langle \\varphi \\rangle_\\theta \\vert ^2\\right)$'
        plt.ylabel(my_ylabel)
        plt.grid(True)
        my_colorlist = plt.cm.plasma(np.linspace(0,1,kx_bar.size))
        my_legend = []
        kxs_to_plot=kx_bar
        for ikx in range(kx_bar.size):
            if kx_bar[ikx] in kxs_to_plot:
                plt.plot(t[0:nt], np.log(phi2[0:nt,ikx]), color=my_colorlist[ikx])
        axes=plt.gca()

        pdfname = 'phi2_vs_t_by_kx'
        pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
        plt.savefig(pdfname)
        
        plt.clf()
        plt.cla()



        # Plot phi2 vs (theta-theta0), one dmid per plot, one series of plots per selected time
        # Plot vs lin-lin, lin-log and log-log scales

        if plot_phi_vs_tt0:

            # lin-lin

            plt.figure(figsize=(12,8))
            it_toPlot = [int(r*(nt-1)) for r in tRatio_toPlot]

            for it in it_toPlot:

                tmp_pdf_id = 1
                pdflist = []
                for idmid in range(len(dmid_list)):
                    if g_exb == 0.0:
                        plt.title('$t=$' + gplot.str_t(t[it]) + ', $k_y=$' + gplot.str_ky(ky) + ', $\\theta_0 =$' + gplot.str_tt0(theta0[itheta0_list[idmid]]))
                    else:
                        plt.title('$t=$' + gplot.str_t(t[it]) + ', $k_y=$' + gplot.str_ky(ky) + ', $\\theta_0^* =$' + gplot.str_tt0(theta0_star[it,itheta0_list[idmid]]))
                    lphi, = plt.plot(bloonang[idmid][it],phi2bloon[idmid][it], marker='o', color=gplot.myblue, \
                            markersize=5, markerfacecolor=gplot.myblue, markeredgecolor=gplot.myblue, linewidth=3.0)
                    lphi.set_label('$\\vert \\varphi \\vert ^2$')
                    lbdry, = plt.plot(bloonang_bndry[idmid][it],phi2bloon_discont[idmid][it], linestyle='', \
                            marker='d', markersize=15, markerfacecolor='r', markeredgecolor='r')
                    lbdry.set_label('_skip')
                    if g_exb == 0.0:
                        plt.xlabel('$\\theta-\\theta_0$')
                    else:
                        plt.xlabel('$\\theta-\\theta_0^*$')
                    plt.grid(True)
                    gplot.legend_matlab()
                    ax = plt.gca()
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
                    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdfname)
                    tmp_pdf_id = tmp_pdf_id+1

                merged_pdfname = 'phi_vs_theta_it_' + str(it)
                gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
                plt.clf()
                plt.cla()

            if plot_phi_vs_tt0_log:

                # lin-log

                plt.figure(figsize=(12,8))
                it_toPlot = [int(r*(nt-1)) for r in tRatio_toPlot]

                for it in it_toPlot:

                    tmp_pdf_id = 1
                    pdflist = []
                    for idmid in range(len(dmid_list)):
                        if g_exb == 0.0:
                            plt.title('$t=$' + gplot.str_t(t[it]) + ', $k_y=$' + gplot.str_ky(ky) + ', $\\theta_0 =$' + gplot.str_tt0(theta0[itheta0_list[idmid]]))
                        else:
                            plt.title('$t=$' + gplot.str_t(t[it]) + ', $k_y=$' + gplot.str_ky(ky) + ', $\\theta_0^* =$' + gplot.str_tt0(theta0_star[it,itheta0_list[idmid]]))
                        lphi, = plt.semilogy(bloonang[idmid][it],phi2bloon[idmid][it], marker='o', color=gplot.myblue, \
                                markersize=5, markerfacecolor=gplot.myblue, markeredgecolor=gplot.myblue, linewidth=3.0)
                        lphi.set_label('$\\vert \\varphi \\vert ^2$')
                        lbdry, = plt.semilogy(bloonang_bndry[idmid][it],phi2bloon_discont[idmid][it], linestyle='', \
                                marker='d', markersize=15, markerfacecolor='r', markeredgecolor='r')
                        lbdry.set_label('_skip')
                        if g_exb == 0.0:
                            plt.xlabel('$\\theta-\\theta_0$')
                        else:
                            plt.xlabel('$\\theta-\\theta_0^*$')
                        plt.grid(True)
                        gplot.legend_matlab()
                        ax = plt.gca()
                        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                        gplot.save_plot(tmp_pdfname, run, ifile)
                        pdflist.append(tmp_pdfname)
                        tmp_pdf_id = tmp_pdf_id+1

                    merged_pdfname = 'phi_vs_theta_linlog_it_' + str(it)
                    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
                    plt.clf()
                    plt.cla()




   # Snapshots and movies

    for idmid in range(len(dmid_list)):
    
        dmid = dmid_list[idmid]

        if (phi_t_present):

            # Snapshots of phi

            if (make_snaps):

                if itSnap_max == -1:
                    itSnap_max = nt

                # find global min and max of ballooning angle
                bloonang_min = 0.
                bloonang_max = 0.
                for it in range(max_it_for_snap):
                    if np.min(bloonang[idmid][it]) < bloonang_min:
                        bloonang_min = np.min(bloonang[idmid][it])
                    if np.max(bloonang[idmid][it]) > bloonang_max:
                        bloonang_max = np.max(bloonang[idmid][it])


                # Snapshots of phi

                plt.figure(figsize=(12,8))

                tmp_pdf_id = 1
                pdflist = []
                for it in range(0,max_it_for_snap,it_step_for_snap):
                    l1, = plt.plot(bloonang[idmid][it],phi2bloon[idmid][it], marker='o', color=gplot.myblue, \
                            markersize=5, markerfacecolor=gplot.myblue, markeredgecolor=gplot.myblue, linewidth=3.0)
                    l1.set_label('$\\vert \\varphi \\vert ^2$')
                    l2, = plt.plot(bloonang_bndry[idmid][it],phi2bloon_discont[idmid][it], linestyle='', \
                            marker='d', markersize=15, markerfacecolor='r', markeredgecolor='r')
                    l2.set_label('_skip')
                    plt.xlabel('$\\theta$')
                    plt.grid(True)
                    plt.gca().set_xlim(bloonang_min,bloonang_max)
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
                    ymin = np.amin(phi2bloon[idmid][it])
                    ymax = np.amax(phi2bloon[idmid][it])
                    ax = plt.gca()
                    ax.set_ylim(ymin,ymax)
                    ax.set_title('$\\theta_0={:.2f},\ $'.format(theta0[itheta0_list[idmid]]) + '$k_y={:.2f},\ t={:.2f}$'.format(ky,t[it]))
                    ax.legend()
                    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                    gplot.save_plot(tmp_pdfname, run, ifile)
                    pdflist.append(tmp_pdfname)
                    tmp_pdf_id = tmp_pdf_id+1

                merged_pdfname = 'snaps_phibloon_dmid_' + str(dmid)
                gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
                plt.clf()
                plt.cla()


            # Movie of phi

            if (make_movies):

                if itMov_max == -1:
                    itMov_max = nt
    
                myfig = plt.figure(figsize=(12,8))

                # find global min and max of ballooning angle
                bloonang_min = 0.
                bloonang_max = 0.
                for it in range(itMov_min, itMov_max):
                    if np.min(bloonang[idmid][it]) < bloonang_min:
                        bAng_min = np.min(bloonang[idmid][it])
                    if np.max(bloonang[idmid][it]) > bloonang_max:
                        bAng_max = np.max(bloonang[idmid][it])
                
                # find phimax for each time step
                phi2_max = np.zeros(nt)
                for it in range(nt):
                    phi2_max[it] = np.max(phi2bloon[idmid][it])
                
                # movie name
                moviename = 'mov_phi_vs_theta' + '_dmid_' + str(dmid)
                moviename = run.out_dir + moviename + '_' + run.fnames[ifile] + '.mp4'
               
                print("\ncreating movie of phi vs ballooning angle ...")

                # intialise artists
                xdata1, ydata1 = [], []
                lphi, = plt.plot([],[], marker='o', color=gplot.myblue, \
                        markersize=5, markerfacecolor=gplot.myblue, markeredgecolor=gplot.myblue, linewidth=3.0)
                lphi.set_label('$\\vert \\varphi \\vert ^2/\\max_{K_x}\\vert\\varphi\\vert^2$')
                xdata2, ydata2 = [], []
                lbdry, = plt.plot([],[], linestyle='', \
                        marker='d', markersize=15, markerfacecolor='r', markeredgecolor='r')
                lbdry.set_label('_skip')

                # Labels, limits, legend
                if g_exb == 0.0:
                    plt.xlabel('$\\theta-\\theta_0$')
                else:
                    plt.xlabel('$\\theta-\\theta_0^*$')
                plt.grid(True)
                plt.gca().set_xlim(bAng_min,bAng_max)
                ax = plt.gca()
                plt.gca().set_ylim(0.0,1.0)
                gplot.legend_matlab()

                # Update lines
                def update_plot(data):

                    # Unpack data from yield_data
                    t, bAng, phi2bAng, bAng_bndry, phi2bAng_discont = data
                    # Update phi2 chain
                    lphi.set_data(bAng,phi2bAng)
                    # Update discontinuities at 2pi interfaces
                    lbdry.set_data(bAng_bndry,phi2bAng_discont)
                    # Update title
                    if g_exb == 0.0:
                        plt.gca().set_title('$t=$' + gplot.str_t(t) + ', $k_y=$' + gplot.str_ky(ky) + ', $\\theta_0 =$' + gplot.str_tt0(theta0[itheta0_list[idmid]]))
                    else:
                        plt.gca().set_title('$t=$' + gplot.str_t(t) + ', $k_y=$' + gplot.str_ky(ky) + ', $\\theta_0^* =$' + gplot.str_tt0(theta0_star[it,itheta0_list[idmid]]))

                    return lphi, lbdry

                # "yield" = "return, and next time function is called, start from there"
                def yield_data():

                    for it in range(itMov_min, itMov_max, itMov_step):

                        sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1))) # comment out on HPC
                        yield t[it], bloonang[idmid][it], phi2bloon[idmid][it]/phi2_max[it], bloonang_bndry[idmid][it], phi2bloon_discont[idmid][it]

                mov = anim.FuncAnimation(myfig, update_plot, frames=yield_data, blit=False, save_count=len(range(itMov_min, itMov_max, itMov_step)))
                writer = anim.writers['ffmpeg'](fps=15,bitrate=1800)
                mov.save(moviename,writer=writer,dpi=100)
                plt.clf()
                plt.cla()

                print("\n... movie completed.")
                print('\n')


            # Movie of discontinuities in phi

            if make_movie_discont:

                if itMov_max == -1:
                    itMov_max = nt

                ## movie of phi2 jump at interfaces between 2pi domains
                moviename = 'phijump' + '_dmid_' + str(dmid)
                moviename = run.out_dir + moviename + '_' + run.fnames[ifile] + '.mp4'

                print("\ncreating movie of phijump vs ballooning angle ...")
                xdata1, ydata1 = [], []
                l1, = plt.plot([],[], marker='o', color=gplot.myred, \
                        markersize=12, markerfacecolor=gplot.myred, markeredgecolor=gplot.myred, linewidth=3.0)
                plt.xlabel('$\\theta$') # NDCPARAM: check for plot_against_theta0_star
                plt.ylabel('$\\Delta\\vert \\varphi \\vert ^2/\\vert \\varphi \\vert ^2$')
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
                    xdata1 = bloonang_bndry[idmid][it]
                    ydata1 = phi2bloon_jump[idmid][it]
                    plt.gca().set_title('$\\theta_0 = {:.2f},$'.format(theta0[itheta0_list[idmid]])+'$k_y={:.2f}, t={:.2f}$'.format(ky,t[it]))
                    l1.set_data(xdata1,ydata1)
                    return l1

                mov = anim.FuncAnimation(myfig,update_mov_jump,init_func=init_mov_jump, \
                        frames=range(0,max_it_for_mov,it_step_for_mov),blit=False)
                writer = anim.writers['ffmpeg'](fps=10,bitrate=-1,codec='libx264')
                mov.save(moviename,writer=writer,dpi=100)
                plt.clf()
                plt.cla()

                print("\n... movie completed.")

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

