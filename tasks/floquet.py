from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as anim
from matplotlib.lines import Line2D
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

from scipy.signal import find_peaks

import gs2_plotting as gplot
import gs2_inputs as inputs

from scipy.special import j0

f_colors = {'phi':'blue', 'apar':'red', 'bpar':'green', 'other':'black'}
#f_labels = {'phi':r'$|\phi|$', 'apar':r'$|A_\parallel|$', 'bpar':r'$|B_\parallel|$'}
f_labels = {'phi':r'|\phi|', 'apar':r'|A_\parallel|', 'bpar':r'|B_\parallel|'}

################################################################################
# A twist and shift chain is identified by it, iky and dmid = 0, ..., min(jtwist-1, (nakx-1)//2)
# which is the number of dkx between kx=0 and the smallest kx>0 that is a member of the chain.
################################################################################

################################################################################
# main
################################################################################
# Plot snapshots of phi along the ballooning chain
#plot_phi_snaps = False
plot_phi_snaps = True

# Plot snapshots of omega
plot_omega_snaps = False
#plot_omega_snaps = True

# Plot single floquet time, or the full simulation?
plot_snap_start = False
#plot_snap_start = True
plot_snap_Tfs = 20
snap_step_multiplier = 1

# When plotting snapshots, fixes the y axes to show the maximum value in the set of frames we are looking at.
y_normalized = True

# Number of floquet periods over which we average for the instantaneous growth-rate.
Nf_avg = 35#-1

def my_task_single(ifile, run, myin, myout, mytime, task_space):  
    
    ##### User's parameters #####
    # We include kxbar=my_dmid*dkx in our chain
    my_dmid=0 

    # Select time index for plot of phi vs ballooning angle
    my_it = [-1]

    ##### End of user's parameters #####
    
    print('Starting single task: Floquet ...\n')

    my_vars = []

    if run.only_plot:
        my_vars = read_from_dat(ifile, run)
    else:
        # Select chain(s)
        iky_list = [i for i in range(1,myout['ky'].size)]
        my_vars = process_and_save_to_dat(ifile, run, myin, myout, mytime, my_dmid, iky_list)
        
    if not run.no_plot:
        plot_task_single(ifile, run, my_vars, my_it, my_dmid)

    print('\n... single task completed.')


def process_and_save_to_dat(ifile, run, myin, myout, mytime, my_dmid, iky_list):

    plot_against_theta0_star = True # TODO check what this does.
    t = myout['t']
    delt = myin['knobs']['delt']
    print('Time step size: {:1.1f}'.format(delt))
    nt = t.size
    nwrite = myin['gs2_diagnostics_knobs']['nwrite']
    print('nwrite: {}'.format(nwrite))

    theta = myout['theta']
    ntheta = theta.size
    theta0 = myout['theta0']
        
    gexb = myin['dist_fn_knobs']['g_exb']
    shat = myin['theta_grid_parameters']['shat']
    jtwist = int(myin['kt_grids_box_parameters']['jtwist'])
        # kx grid in gs2
    kx_gs2 = myout['kx']
    # ky grid in gs2
    ky = myout['ky']
    # Spacing in grids. 
    dky = 1./myin['kt_grids_box_parameters']['y0']
    dkx = 2.*pi*abs(shat)*dky/jtwist
    nakx = kx_gs2.size
    naky = ky.size
    # kx grid has not been sorted to be monotonic yet, so maximum occurs (approximately) halfway.
    ikx_max = int(round((nakx-1)/2))
    ikx_min = ikx_max+1
    connections = (nakx + 1)/jtwist
    
    # Terms for kperp2 and driving drifts.
    gds2 = myout['gds2']
    gds21 = myout['gds21']
    gds22 = myout['gds22']
    gbdrift = myout['gbdrift']
    gbdrift0 = myout['gbdrift0']
    cvdrift = myout['cvdrift']
    cvdrift0 = myout['cvdrift0']

    
    # Floquet period and number of t-steps in Floquet period.
    # and the number of t-steps before ExB remap.
    if gexb != 0.0:
        Tf = abs(2*pi*shat/gexb)
        Nf = int(round(Tf/delt))
        N = max(1, abs(int(round(dkx/(gexb*delt*dky)))))
        shexb_sign = 1# int(shat/gexb/abs(shat/gexb))
        print('Floquet period: {:1.1f}'.format(Tf))
        print('Number of t-steps in Floquet period: {}'.format(Nf))
        print('Number of t-steps before ExB re-map: {}'.format(N))
    else:
        Tf = np.nan
        Nf = np.nan
        N = np.nan

    phi2_gs2 = myout['phi2_by_mode'][:,:,:]         # phi squared [t, ky, kx]
    
    if myout['omega_average_present']:
        omega_gs2 = myout['omega_average'][:,:,:,0]     # Real frequency [t, ky, kx]

    fields_present = []
    fields_by_mode = []
    
    if myout['phi2_by_mode_present']:
        fields_by_mode.append(myout['phi2_by_mode'])
        fields_present.append('phi')
    if myout['apar2_by_mode_present']:
        fields_by_mode.append(myout['apar2_by_mode'])
        fields_present.append('apar')
    if myout['bpar2_by_mode_present']:
        fields_by_mode.append(myout['bpar2_by_mode'])
        fields_present.append('bpar')
    
    fields_by_mode = np.sqrt(np.asarray(fields_by_mode))

    fields_t_present = []
    fields_t = []
    
    if myout['phi_t_present']:
        fields_t.append(myout['phi_t'])
        fields_t_present.append('phi')
    if myout['apar_t_present']:
        fields_t.append(myout['apar_t'])
        fields_t_present.append('apar')
    if myout['bpar_t_present']:
        fields_t.append(myout['bpar_t'])
        fields_t_present.append('bpar')
    fields_t = np.asarray(fields_t)   # [field, t, ky, kx, theta, ri]
        
    if not len(fields_t_present) > 0:
        print("No fields pver time present for analysis. Qutting.")
        quit()

    # sorting kx_gs2 to get monotonic kx_bar
    kx_bar = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))

    fields_t_abs = np.sqrt(np.sum(np.power(fields_t, 2), axis=5, keepdims=True))       # absolute value of fields squared, with theta included [field, t, ky, kx, theta]
    fields_t = np.concatenate(( fields_t, fields_t_abs ), axis = 5)
    fields_t = np.concatenate(( fields_t[:,:,:,ikx_min:,:,:], fields_t[:,:,:,:ikx_min,:,:] ), axis=3)   # [field, t, ky, kx, theta, ria])

    fields_by_mode = np.concatenate(( fields_by_mode[:,:,:,ikx_min:], fields_by_mode[:,:,:,:ikx_min] ), axis=3 )

    if myout['omega_average_present']:
        omega = np.concatenate((omega_gs2[:,:,ikx_min:], omega_gs2[:,:,:ikx_min]), axis=2)
    else:
        omega = None

    # get kx and kx_star from kx_bar
    kx = np.zeros((nt,naky,nakx))
    kx_star = np.zeros((nt,naky,nakx))
    # @ it = 0, kx = kx_bar
    # First step is taken with 0.5*dt
    # Other steps taken with full dt
    for it in range(1,nt):
        for iky in range(naky):
            ikx_shift = int(round(gexb*ky[iky]*delt*(nwrite*it-0.5)/dkx))
            for ikx in range(nakx):
                kx[it,iky,ikx] = kx_bar[ikx] + ikx_shift*dkx
                kx_star[it,iky,ikx] = kx[it,iky,ikx] - gexb*ky[iky]*delt*(nwrite*it-0.5)

    iky = 1
    ikx = 1
    it = 52
    dikx = -1#jtwist*iky
    print(t[it])
    x = (gexb*ky[iky]*(t-0.5*delt)/dkx)[1:it]
    #y = (kx_star[1:it,iky,ikx] + kx_bar[ikx])/dkx
    y = kx_star[1:it,iky,ikx]
    #ydikx = (kx_star[1:it,iky,ikx + dikx] + kx_bar[ikx+dikx])/dkx
    ydikx = kx_star[1:it,iky,ikx + dikx]
    th0 = y*dkx/(shat*ky[iky])
    plt.plot(x,y,label=['ikx=1'])
    plt.plot(x,ydikx,label='-dikx')
    print(kx_bar[ikx])
    print(kx_bar[ikx+dikx])
    plt.grid()
    plt.legend()
    plt.savefig('test.pdf')
    #quit()
    # index of kx=0
    ikx0 = (nakx-1)//2
    
    # In the following :
    # -- compute ballooning angle and store in bloonang_chain[it][ibloon]
    # -- construct associated phi2 and store in phi2bloon_chain[it][ibloon]
    # -- compute associated growthrate gamma, such that :
    #        phinew = phi * exp(-i*omega*delt)
    #        gamma = Im(omega)
    #
    
    ikx_members = []
    ikx_prevmembers = []
    bloonang = []
    kperp = []
    drift = []
    bloonang_bndry = []
    fieldsbloon = []

    # For each ky, the maximum value of fields along the ballooning chain, as a function of time. [ky,t]
    max_fieldsbloon = []
    omegabloon = []
    kxbarbloon = []
    gamma = []

    for iky in iky_list:

        ikx_members_chain = []
        ikx_prevmembers_chain = []
        bloonang_chain = []
        kperp_chain = []
        drift_chain = []
        bloonang_bndry_chain = []
        fieldsbloon_chain = []
        max_fieldsbloon_chain = []
        omegabloon_chain = []
        kxbarbloon_chain = []

        iTf = 0

        # Initialize ikx_shift and ikx_shift_old that are used in the following loop.
        ikx_shift = 0
        ikx_shift_old = 0
       
        if shat > 0:
            shat_sign = 1
        else:
            shat_sign = -1

        for it in range(nt):
            ikx_members_now = []                # The ikxs that are included in the chain at t-step it.
            ikx_prevmembers_now = []            # The ikxs that were included in the chain at the last t-step (it-1), now.
            bloonang_now = []                   
            kperp_now = []                   
            drift_now = []                   
            bloonang_bndry_now = []
            fieldsbloon_now = []
            omegabloon_now = []
            kxbarbloon_now = []

            ikx_shift_old = ikx_shift
            if it > 0:
                ikx_shift = int(round(gexb*ky[iky]*delt*(nwrite*it-0.5)/dkx))
            
            # Build ikx_members_now and ikx_prevmembers_now.

            # Get all members of ballooning chain.
            # Move to the kx chain that contains kxstar(t=0) = my_dmid * dkx.
            ikx = ikx0 - ikx_shift + my_dmid
            ikxprev = ikx0 - ikx_shift_old + my_dmid
            
            # Now move right in the chain until we are as far right as possible without being greater than nakx.
            while(ikx < nakx - jtwist*iky):
                ikx = ikx + jtwist*iky
                ikxprev = ikxprev + jtwist*iky
            
            # Note that if ikx_shift is negative and large (requires gexb<0), ikx0 - ikx_shift may start off already > nakx (off the end of the grid we want to work back from)
            while(ikx >= nakx):
                ikx = ikx - jtwist*iky
                ikxprev = ikxprev - jtwist*iky

            # Then, until we reach the end of the chain, add the kx to the chain and calculate the total phi2 along the chain.
            while(ikx >= 0):
                if ikxprev >= nakx or ikxprev < 0:
                    ikx_prevmembers_now.append(np.nan)
                else:
                    ikx_prevmembers_now.append(ikxprev)
                ikx_members_now.append(ikx)
                ikx = ikx - jtwist*iky
                ikxprev = ikxprev - jtwist*iky

            # Calculate and store the fields along the ballooning chain with all theta. 
            
            # This is the indices of members in this chain.
            member_range = range(0, len(ikx_members_now), 1)                
            
            # Compute ballooning angle and construct associated field(s).
            for imember in member_range:
                # Different definition of the ballooning angle depending on whether we choose kx_star or kx.
                if plot_against_theta0_star:
                    theta0_choice = kx_star[it,iky,ikx_members_now[imember]]/(shexb_sign*shat*ky[iky])
                else:
                    theta0_choice  = kx[it,iky,ikx_members_now[imember]]/(shexb_sign*shat*ky[iky])

                if shat > 0:
                    theta_range = range(0, ntheta-1, 1)
                else:
                    theta_range = range(ntheta-1, 0, -1)

                for itheta in theta_range:
                    bloonang_now.append( theta[itheta] - theta0_choice )
                    kperp_now.append( np.sqrt( ky[iky]**2 * (gds2[itheta] + 2*theta0_choice*gds21[itheta] + theta0_choice**2 * gds22[itheta]) ) )
                    drift_now.append( ky[iky] * (gbdrift[itheta] + cvdrift[itheta] + theta0_choice*(gbdrift0[itheta] + cvdrift0[itheta])) )
                    fieldsbloon_now.append(fields_t[:,it,iky,ikx_members_now[imember],itheta,:])
                kxbarbloon_now.append(kx_bar[ikx_members_now[imember]])
                
                # Construct chain of real frequency
                if omega is not None:
                    omegabloon_now.append(omega[it,iky,ikx_members_now[imember]])

                # Saving bloonang at boundary of modes with different kx.
                member_range = range(len(ikx_members_now)-1)
                for imember in member_range:
                    b_ang_bndry = shat_sign*pi-kx_star[it,iky,ikx_members_now[imember]]/(shat*ky[iky])
                    bloonang_bndry_now.append(b_ang_bndry)
            ikx_members_chain.append(ikx_members_now)
            ikx_prevmembers_chain.append(ikx_prevmembers_now)
            bloonang_chain.append(bloonang_now)
            kperp_chain.append(kperp_now)
            drift_chain.append(drift_now)
            bloonang_bndry_chain.append(bloonang_bndry_now)
            fieldsbloon_chain.append(fieldsbloon_now)
            omegabloon_chain.append(omegabloon_now)
            kxbarbloon_chain.append(kxbarbloon_now)
            max_fieldsbloon_chain.append(np.amax(fieldsbloon_now, axis=0))
    
        # Adding the chain to the full collection
        ikx_members.append(ikx_members_chain)
        ikx_prevmembers.append(ikx_prevmembers_chain)
        bloonang.append(bloonang_chain)
        kperp.append(kperp_chain)
        drift.append(drift_chain)
        bloonang_bndry.append(bloonang_bndry_chain)
        fieldsbloon.append(fieldsbloon_chain)
        max_fieldsbloon.append(max_fieldsbloon_chain)
        omegabloon.append(omegabloon_chain)
        kxbarbloon.append(kxbarbloon_chain)

    # Now we compute the two types of linear growth rate: 
    # - Time average over Floquet behaviour;
    # - Instantaneous maximum during the final Floquet period.

    # We start looking over a time-window that begins at t[it_start].
    if gexb != 0.0:
        if Nf_avg == -1:
            N_from_end  = int(mytime.twin * Nf)-1
        else:
            N_from_end = Nf_avg
        it_start = max(0,nt-int(round((N_from_end*Tf/delt)/nwrite)))
    else:
        # Uses mytime.twin to get the starting time. If mytime.twin=0.1, then we only look at the last [(1-0.1)]*100% of the data.
        it_start = round(mytime.twin*nt)

    t_collec = []
    max_fields_collec = []
    
    # Compute the time-average. Will end up with a linear fit that consists of a slope and offset (y=slope*x+offset)
    # Store the maximum slope and offset.
    slope_max = np.zeros(len(iky_list))
    offset_max = np.zeros(len(iky_list))
        
    # [iky, t, field, ria]
    for iiky in range(len(iky_list)):
        max_fields_tmp = np.zeros( ( len(max_fieldsbloon[iiky])-it_start, len(max_fieldsbloon[0][0]) ) )
        for it in range(max_fields_tmp.shape[0]):
            for ifield in range(max_fields_tmp.shape[1]):
                max_fields_tmp[it,ifield] = max_fieldsbloon[iiky][it_start+it][ifield][2]
        if it_start > 0:
            max_fields_tmp = max_fields_tmp/max_fields_tmp[0,:]      # Normalize to field value at start time.
        max_fields_collec.append(max_fields_tmp)
        t_tmp = np.zeros(len(t)-it_start)
        for it in range(t_tmp.size):
            t_tmp[it] = t[it_start+it]
        t_collec.append(t_tmp)

        # Just get growth rates from first field - should settle in to same growth rates for all fields anyway.
        [gam,offset] = leastsq_lin(t_tmp,np.log(max_fields_tmp[:,0]))
        slope_max[iiky] = gam
        offset_max[iiky] = offset

    # At this point:
    # fit_avg(t) ~ field(tstart) * exp[gam*t + offset]

    gamma_max = np.zeros(len(iky_list))
    gamma_max_std = np.zeros(len(iky_list))
    it_gamma_max = np.zeros(len(iky_list))
    if gexb != 0.0:
        # Number of Floquet periods back we look for maxima. User defined or could be modified.
        Nf_for_search = N_from_end//2
        # For each floquet period that we are looking at:
        for iiky in range(len(iky_list)):
            gamma_max_per_nf = np.zeros(Nf_for_search)
            it_gamma_max_per_nf = np.zeros(Nf_for_search)
            # Compute the instantaneous maximum.
            # Compute gamma_max from ln(field_max)
            for iF in range(Nf_for_search):
                # Start looking for derivatives one Floquet period before last time-step
                it_end_nf = int( len(max_fieldsbloon[iiky])-1 - iF*Tf//(delt*nwrite) )
                it_start_nf = int(it_end_nf - Tf//(delt*nwrite))
                for it in range(it_start_nf, it_end_nf):
                    gamma_max_tmp = ( np.log(max_fieldsbloon[iiky][it+1][0][2]) - np.log(max_fieldsbloon[iiky][it-1][0][2]) ) / (2*delt*nwrite)
                    if (gamma_max_tmp > gamma_max_per_nf[iF]):
                        it_gamma_max_per_nf[iF] = it
                        gamma_max_per_nf[iF] = gamma_max_tmp
            it_gamma_max_per_nf = it_gamma_max_per_nf.astype(int)
            
            # Now we have several instantaneous maxima. 
            # Calculate the average, and standard deviation.
            gamma_max[iiky] = (max(gamma_max_per_nf)+min(gamma_max_per_nf))/2.
            gamma_max_std[iiky] = max(gamma_max_per_nf)-gamma_max[iiky]
            # Get the gradient which is closest to the average.
            it_gamma_max[iiky] = int(it_gamma_max_per_nf[ int((np.abs(gamma_max_per_nf-gamma_max[iiky])).argmin()) ])
            
        # At this point:
        # fit_max(t) ~ phi(tstart) * exp[gamma_max*(t-t_gamma_max)]

    else:
        # If we have no flow shear, then we just do a normal fit to an exponential.
        it_start = round(0.5*nt)
        gamma = np.zeros((nakx,len(iky_list)))
        tofit = np.amax(fields_t,axis=4)
        for ikx in range(nakx):
            for iiky in range(len(iky_list)):
                iky = iky_list[iiky]
                gamma[ikx,iiky] = get_growthrate(t,tofit_sq[0,:,iky,ikx,2],it_start)
    it_gamma_max = it_gamma_max.astype(int)

    if 'omprimfac' in myin['dist_fn_knobs']:
        omprimfac = myin['dist_fn_knobs']['omprimfac']
    else:
        omprimfac = 1.0
    qinp = myin['theta_grid_parameters']['qinp']
    beta = myin['parameters']['beta']

    # Saving variables to mat-file
    my_vars = {}
    my_vars['Nf'] = Nf
    my_vars['t'] = t
    my_vars['delt'] = delt
    my_vars['nwrite'] = nwrite
    my_vars['shat'] = shat
    my_vars['gexb'] = gexb
    my_vars['kx'] = kx
    my_vars['ky'] = ky
    my_vars['dkx'] = dkx
    my_vars['connections'] = connections
    my_vars['jtwist'] = jtwist
    my_vars['iky_list'] = iky_list
    my_vars['nakx'] = nakx
    my_vars['kx_bar'] = kx_bar
    my_vars['kx_star'] = kx_star
    my_vars['ikx_members'] = ikx_members
    my_vars['ikx_prevmembers'] = ikx_prevmembers
    my_vars['bloonang'] = bloonang
    my_vars['bloonang_bndry'] = bloonang_bndry
    my_vars['kperp'] = kperp
    my_vars['drift'] = np.asarray(drift)
    my_vars['omegabloon'] = omegabloon
    my_vars['kxbarbloon'] = kxbarbloon
    my_vars['gamma'] = gamma
    my_vars['gamma_avg'] = slope_max
    my_vars['gamma_avg_offset'] = offset_max
    my_vars['gamma_max'] = gamma_max
    my_vars['gamma_max_std'] = gamma_max_std
    my_vars['it_gamma_max'] = it_gamma_max
    my_vars['it_start'] = it_start
    my_vars['fields_present'] = fields_present
    my_vars['fields_by_mode'] = fields_by_mode
    my_vars['fields_t_present'] = fields_t_present
    my_vars['fields_t'] = fields_t
    my_vars['fieldsbloon'] = fieldsbloon
    my_vars['max_fieldsbloon'] = max_fieldsbloon
    my_vars['max_fields_collec'] = max_fields_collec
    my_vars['t_collec'] = t_collec
    my_vars['qinp'] = qinp
    my_vars['omprimfac'] = omprimfac
    my_vars['beta'] = beta
    my_vars['maxchainlength'] = (nakx-1)/(jtwist*(len(ky)-1))
    my_vars['dtheta0'] = [ 1.0/(i*jtwist) for i in iky_list ]
    print(iky_list)
    my_vars['input_file'] = myin

    datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.floquet.dat'
    with open(datfile_name, 'wb') as outfile: # 'wb' stands for write bytes
        pickle.dump(my_vars,outfile)

    return my_vars


#################### Read Floquet variables from dat file ####################

def read_from_dat(ifile, run):


    datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.floquet.dat'
    with open(datfile_name, 'rb') as datfile: # 'rb' stands for read bytes
        return pickle.load(datfile)

#################### Plotting ####################
    
def plot_task_single(ifile, run, my_vars, my_it, my_dmid):
    
    ##### User variables #####
    skip_init = False
    
    Nf = my_vars['Nf']
    t = my_vars['t']
    delt = my_vars['delt']
    nwrite = my_vars['nwrite']
    shat = my_vars['shat']
    gexb = my_vars['gexb']
    kx = my_vars['kx']
    ky = my_vars['ky']
    dkx = my_vars['dkx']
    jtwist = my_vars['jtwist']
    connections = my_vars['connections']
    iky_list = my_vars['iky_list']
    nakx = my_vars['nakx']
    kx_bar = my_vars['kx_bar']
    kx_star = my_vars['kx_star']
    fields_present = my_vars['fields_present']
    fields_by_mode = my_vars['fields_by_mode']
    fields_t_present = my_vars['fields_t_present']
    fields_t = my_vars['fields_t']
    fieldsbloon  = my_vars['fieldsbloon']
    max_fieldsbloon  = my_vars['max_fieldsbloon']
    max_fields_collec = my_vars['max_fields_collec']    
    t_collec = my_vars['t_collec']    
    ikx_members = my_vars['ikx_members']
    ikx_prevmembers = my_vars['ikx_prevmembers']
    bloonang = my_vars['bloonang']
    bloonang_bndry = my_vars['bloonang_bndry']
    kperp = my_vars['kperp']
    drift = np.asarray(my_vars['drift'])
    omegabloon = my_vars['omegabloon']
    kxbarbloon = my_vars['kxbarbloon']
    gamma = my_vars['gamma']
    slope_max = my_vars['gamma_avg']
    offset_max = my_vars['gamma_avg_offset'] 
    gamma_max = my_vars['gamma_max']
    gamma_max_std = my_vars['gamma_max_std']
    it_gamma_max = my_vars['it_gamma_max']
    it_start = my_vars['it_start']

    Tf = Nf*delt
    nt = t.size

    print("Beginning plot for Floquet modes")

    myfig = plt.figure(figsize=(12,8))
    
    # Plot sum and max of phi2 vs time for every ky
    plt.figure(figsize=(12,8))
    plt.xlabel('$t$')
    
    my_legend = []
    if gexb != 0.0 and t_collec is not None:
        
        # Growth rate spectrum if more than 2 kys.
        if len(ky)>2:
            fig, axes = plt.subplots(nrows=2, figsize=(16,16), sharex=True)
            print(ky)
            gplot.plot_1d(ky[1:], slope_max, axes=axes[0], xlab='', ylab=r'$\langle{\gamma}\rangle_t a_{GS2}/v_{th,ref}$', marker='s')
            gplot.plot_1d(ky[1:], gamma_max, axes=axes[1], xlab='$k_y$',ylab=r'$\gamma_{\rm{max}}a_{GS2}/v_{th,ref}$', errors=gamma_max_std, marker='s' )
            pdfname = 'ky_scan' 
            pdfname = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
            fig.savefig(pdfname)

        print("Plotting maximum and average linear growth-rates.")
        fig = plt.figure(figsize=(12,8))
        ax = plt.gca()
        if skip_init:
            for iiky in range(len(iky_list)):
                gplot.plot_1d(t_collec[iiky], max_fields_collec[iiky][0], '$t$', axes = ax, color = '#777777', linewidth=1.0, log='y')
                my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[iiky]]))
                gplot.plot_1d(t_collec[iiky], np.exp(slope_max[iiky]*t_collec[iiky]+offset_max[iiky]), '$t$', axes = ax, color = '#00aacc', linewidth=1.0, log='y', linestyle='--')
                my_legend.append('$\\langle\\gamma\\rangle_t = {:.3f}$'.format(slope_max[iiky]))
                bot, top = plt.ylim()
                gplot.plot_1d(t_collec[iiky], max_fieldsbloon[iiky][it_gamma_max[iiky]]/max_fieldsloon[iiky][it_start] * np.exp(gamma_max[iiky]*(t_collec[iiky]-t[it_gamma_max[iiky]])), '$t$', axes = ax, color = '#ff8c40', linewidth=1.0, log='y', linestyle=':')
                my_legend.append(r"$\gamma_{{\rm{{max}}}} = {:1.3f} \pm {:1.3f} $".format(gamma_max[iiky], gamma_max_std[iiky]))
                plt.ylim(bot,top)
        else:
            for iiky in range(len(iky_list)):
                y = np.asarray(max_fieldsbloon)[iiky,:,0,2]
                gplot.plot_1d(t, y, '$t$', axes=ax, color = '#777777', linewidth=1.0, log='y')
                my_legend.append('$k_y = {:.3f}$'.format(ky[iky_list[iiky]]))
                gplot.plot_1d(t, y[it_start]*np.exp(slope_max[iiky]*t+offset_max[iiky]), '$t$',axes = plt.gca(), color = '#00aacc', linewidth=1.0, log='y', linestyle='--')
                my_legend.append(r'$\langle\gamma\rangle_t = {:.3f}$'.format(slope_max[iiky]))
                bot, top = ax.get_ylim()
                gplot.plot_1d(t, y[it_gamma_max[iiky]]*np.exp(gamma_max[iiky]*(t-t[it_gamma_max[iiky]])), '$t$',axes = plt.gca(), color = '#ff8c40', linewidth=1.0, log='y', linestyle='--')
                my_legend.append('$\\gamma_{max} '+'= {:.3f}$'.format(gamma_max[iiky]))
                ax.set_ylim(bot,top)
        plt.ylabel(r'$\max_{{k_x}}\vert \langle {} \rangle_\theta \vert ^2$'.format( f_labels[fields_present[0]] ))
        plt.legend(my_legend)
        pdfname = 'floquet_max_vs_t_all_ky' + '_dmid_' + str(my_dmid) 
        pdfname = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
        plt.savefig(pdfname)
        plt.clf()
        plt.cla()
        plt.close()
        print("Done.")
      
    print("Plotting phi2 vs t for each kx.")
    for iiky in range(0,len(iky_list)):
        iky = iky_list[iiky]

        # plot phi2 vs t for each kx
        plt.title('$k_y={:.2f}$'.format(ky[iky]))
        plt.xlabel(r'$t\ [r_r/v_{thr}]$')
        my_ylabel = r'$\ln \left( \langle {} \rangle_\theta ^2\right)$'.format( f_labels[fields_present[0]] )
        plt.ylabel(my_ylabel)
        plt.grid(True)
        my_colorlist = plt.cm.plasma(np.linspace(0,1,kx_bar.size))
        my_legend = []
        kxs_to_plot=kx_bar
        for ikx in range(kx_bar.size):
            if kx_bar[ikx] in kxs_to_plot:
                plt.plot(t, np.log(fields_by_mode[0,:,iky,ikx]), color=my_colorlist[ikx])
                #my_legend.append('$\\rho_i\\bar{k}_x = '+str(kx_bar[ikx])+'$')
        #plt.legend(my_legend)
        axes=plt.gca()

        pdfname = '{}_by_kx_iky_{:d}'.format(fields_present[0], iky)
        pdfname = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
        plt.savefig(pdfname)
        
        plt.clf()
        plt.cla()

        ## set up time stepping for snapshots and movies
        # Just one floquet oscillation.
        if plot_snap_start:
            it_start_for_snap = 0
            max_it_for_snap = int(min(nt,plot_snap_Tfs*Nf/nwrite))
        else:
            it_start_for_snap = nt - int(plot_snap_Tfs*Nf/(nwrite))
            max_it_for_snap = nt
        it_step_for_snap = snap_step_multiplier#*int(nwrite) # Adapt this
        if plot_omega_snaps:
            print("Plotting snapshots of omega vs t for each kxbar between it={} and {}".format(it_start_for_snap, max_it_for_snap))
            ## Plot real frequency of connected chain, vs kxbar
            # Save snapshots
            tmp_pdf_id = 1
            pdflist = []
            frame = 1
            n_frames = (max_it_for_snap - it_start_for_snap)//it_step_for_snap + min((max_it_for_snap - it_start_for_snap)%it_step_for_snap,1)
            for it in range(it_start_for_snap,max_it_for_snap,it_step_for_snap):
                print("Snapshot {}/{}...".format(frame,n_frames), end="\r")
                #l1, = plt.plot(kxbarbloon[iiky][it],omegabloon[iiky][it], marker='o', color='#0000ff', \
                #        markersize=12, markerfacecolor='#0000ff', markeredgecolor='#0000ff', linewidth=3.0)
                plt.figure(figsize=(12,8))
                plt.grid(True)
                ax = plt.gca()
                gplot.plot_1d(kxbarbloon[iiky][it], omegabloon[iiky][it], r'$\rho\bar{k}_x$', axes=ax, marker='s')
                plt.ylabel('$\\omega$'+' '+'$[v_{thr}/r_r]$')
                ax.set_title('$k_y={:.2f}, t={:.2f}$'.format(ky[iky],t[it]))
                tmp_pdfname = 'tmp'+str(tmp_pdf_id)
                gplot.save_plot(tmp_pdfname, run, ifile)
                pdflist.append(tmp_pdfname)
                tmp_pdf_id = tmp_pdf_id + 1
                frame = frame + 1

            merged_pdfname = 'omega_snaps'
            gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)
            print("Snapshots merged")
            plt.clf()
            plt.cla()

        if len(fields_t_present)>0 and plot_phi_snaps and run.make_movies:
            print('At least one field_over_t present: Plotting snapshots of fields vs ballooning angles.')
            # find global min and max of ballooning angle
            bloonang_min = 0.
            bloonang_max = 0.
            for it in range(max_it_for_snap):
                if np.min(bloonang[iiky][it]) < bloonang_min:
                    bloonang_min = np.min(bloonang[iiky][it])
                if np.max(bloonang[iiky][it]) > bloonang_max:
                    bloonang_max = np.max(bloonang[iiky][it])
            
            ## Save snapshots
            tmp_pdf_id = 1
            pdflist = []
            height_ratios = [1]*(3+len(fields_t_present))
            height_ratios[-2] = 0.6
            fig, axes = plt.subplots(nrows = len(fields_t_present)+3, figsize = (16, 8*len(fields_t)), gridspec_kw = {'wspace':0, 'hspace':0.0, 'height_ratios':height_ratios} )
            axes[-1].set_yscale('log')
            axes[-2].set_visible(False)
            # Set up formatting for the movie files
            Writer = anim.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            
            lines = []

            # intialize three line objects (one in each axes)
            lines.append(axes[0].plot([], [], lw=1, color='green', linestyle='--', label = r'$J_0(k_\perp)$')[0])
            lines.append(axes[0].plot([], [], lw=1, color='red', linestyle='--', label = r'$\omega_D/\omega_{D,\rm{max}}$')[0])
            lines.append(axes[-1].plot([], [], lw=2, color='black', linestyle = ':')[0])

            for ifield in range(len(fields_t_present)):
                lines.append(axes[ifield+1].plot([], [], lw=2, color=f_colors[fields_t_present[ifield]])[0])
                lines.append(axes[ifield+1].plot([], [], lw=0.5, color=f_colors[fields_t_present[ifield]], linestyle = ':')[0])
                lines.append(axes[ifield+1].plot([], [], lw=0.5, color=f_colors[fields_t_present[ifield]], linestyle = '--')[0])
                lines.append(axes[ifield+1].plot([], [], linestyle="None", color='black', marker='s')[0])
                axes[ifield+1].set_xlim(bloonang_min,bloonang_max)
                lines.append(axes[-1].plot(t, np.array(max_fieldsbloon[iiky])[:,ifield,2], lw=2, color=f_colors[fields_t_present[ifield]], label = r'${}$'.format(f_labels[fields_t_present[ifield]]))[0])
           
            # the same axes initalizations as before (just now we do it for both of them)
            axes[0].set_xlim(bloonang_min,bloonang_max)
            axes[-1].set_xlim(t[it_start_for_snap], t[max_it_for_snap-1])
            print(t[it_start_for_snap])
            print(t[max_it_for_snap-1])
            axes[0].set_ylim(-1, 1)
            max_fieldsbloon_this_t = np.array(max_fieldsbloon[iiky])[ it_start_for_snap:max_it_for_snap, :, 2 ]
            axes[-1].set_ylim(np.amin(max_fieldsbloon_this_t)*0.9, np.amax(max_fieldsbloon_this_t)*1.1)
            for ifield in range(len(fields_t_present)):
                axes[ifield+1].set_ylabel( r'${}$'.format(f_labels[fields_t_present[ifield]] ))
            axes[-1].set_ylabel(r'$\rm{max}(\vert{field}\vert)$')
            axes[-3].set_xlabel(r'$\theta-\theta_0^*$')
            axes[-1].set_xlabel(r'$t$')
            for ax in axes:
                ax.grid()
            def data_gen(theta_dat, kperp_dat, drift_dat, fields_dat, fieldsmax_dat, t_dat, t_range, bloonang_bndry):
                frame = 0
                for it in t_range:
                    print("Frame {}/{}...".format(frame,n_frames), end="\r")
                    theta = theta_dat[it][:]
                    drift = drift_dat[it][:]/np.amax(drift_dat[it][:])
                    kperp = kperp_dat[it][:]
                    fields = np.array(fields_dat[it])
                    t = t_dat
                    this_t = t_dat[it]
                    this_fieldsmax = t_dat[it]
                    bloon_bounds = bloonang_bndry[it][:]
                    frame = frame + 1
                    # adapted the data generator to yield both sin and cos
                    yield theta,kperp,drift,fields,t,this_t,this_fieldsmax,bloon_bounds

            def runplot(data):
                # update the data
                theta, kperp, drift, fields, t, this_t, this_fieldsmax, bloon_bounds = data

                axes[0].figure.canvas.draw()

                # axis limits checking. Same as before, just for both axes
                for ifield in range(len(fields[0])):
                    ylim = np.amax(fields[:, ifield, 2]*1.1)
                    axes[ifield+1].set_ylim(-ylim, ylim)
                    lines[3+ifield*5 + 0].set_data(theta, fields[:,ifield,2])
                    lines[3+ifield*5 + 1].set_data(theta, fields[:,ifield,0])
                    lines[3+ifield*5 + 2].set_data(theta, fields[:,ifield,1])
                    lines[3+ifield*5 + 3].set_data(bloon_bounds, np.zeros(len(bloon_bounds)))
                    axes[ifield+1].figure.canvas.draw()

                # update the data of both line objects
                lines[0].set_data(theta, j0(kperp))
                lines[1].set_data(theta, drift)
                lines[2].set_data([this_t,this_t], [0,np.amax(this_fieldsmax)])
                
                """
                pos = find_peaks(phi2, height=this_phi2max/20)[0]
                maxima_values = np.array(phi2)[pos]
                maxima_theta = np.array(theta)[pos]
                i = 0
                while i < len(pos) and i < len(line)-6:
                    line[6+i].set_data( [maxima_theta[i], maxima_theta[i]], [this_phi2max*2, maxima_values[i]] )
                    i = i + 1
                while i < len(line) - 6:
                    line[6+i].set_data([],[])
                    i = i + 1
                """
                return lines

            # initialize the data arrays 
            n_frames = (max_it_for_snap - it_start_for_snap)//it_step_for_snap + min((max_it_for_snap - it_start_for_snap)%it_step_for_snap,1)
            ani = anim.FuncAnimation(fig, runplot, 
                data_gen( bloonang[iiky], kperp[iiky], np.asarray(drift[iiky]), fieldsbloon[iiky], np.asarray(max_fieldsbloon[iiky]), t, 
                range(it_start_for_snap,max_it_for_snap,it_step_for_snap), bloonang_bndry[iiky]), blit=False, interval=15,
                repeat=False, save_count=n_frames)
            #ani.save('phibloon_t.mp4', writer=writer)
            gplot.save_anim(ani, writer, 'phibloon_t_iiky_{}'.format(iiky), run, ifile = ifile)


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

scanlabels={'ky':r'$k_y$', 'delt':r'$\Delta{t}', 'jtwist':r'jtwist', 'connections':r'connections', 'gexb':r'$\gamma_{E\times B}$', 'beta':r'$\beta$'}
# Function for scanning in the parameter x; x will often be ky, but it could be something else, for example if we are scanning in another parameter at a fixed ky.
def scan1d(run, x):
    # Only execute if plotting
    if run.no_plot:
        return

    print("Plotting scan of growth rates with " + str(x) + "...")
    Nfile = len(run.fnames)

    # Init arrays of data used in scan.
    full_floquet = [dict() for ifile in range(Nfile)]
    
    # Get lingrowth data from .dat file.
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.floquet.dat'
        with open(datfile_name,'rb') as datfile:
            full_floquet[ifile] = pickle.load(datfile)
    
    xdat = np.zeros(Nfile)

    for ifile in range(Nfile):
        lastcharx = x[-1]
        xstem = x[:len(x)-1]
        if xstem in ['dens','temp','fprim','tprim','vnewk']:
            if lastcharx == 'i':
                specindex = 1
            if lastcharx == 'e':
                specindex = 2
            if lastcharx == 'x':
                specindex = 3
            xdat[ifile] = full_floquet[ifile]['specparams'][specindex-1][xstem]
        elif x in ['dens','temp','fprim','tprim','vnewk']:
            xdat[ifile] = full_floquet[ifile]['specparams'][0][x]
        elif x == 'ky':
            # We assume that there is only a single ky in the list
            xdat[ifile] = full_floquet[ifile]['ky'][full_floquet[ifile]['iky_list'][0]]
        elif x == 'dtheta0':
            print(full_floquet[ifile][x])
            xdat[ifile] = full_floquet[ifile][x][-1]
        else:
            if x in full_floquet[ifile]:
                xdat[ifile] = full_floquet[ifile][x]
            else:
                xdat[ifile] = full_floquet[ifile]['input_file'][inputs.nml[x]][x]

    # Make x axis monotonic.
    #xvals = sorted(list(set(xdat)))
    xvals = sorted(xdat)
    sorted_file_indices = [x for _, x in sorted(zip(xdat, range(Nfile)))]
    print(str(x) + " values: " + str(xvals))
    
    # Have collected x-axis values.
    # Now we can show various quantities that vary with the scan:
    # - Time-averaged growth rate. 
    # - Maximum instantaneous growth rate.
    # - Error on instantaneous growth rate.
    # - ...
    
    print(sorted_file_indices)

    gammatavgs = np.zeros(len(xvals))
    gammainsts = np.zeros(len(xvals))
    gammainst_stds = np.zeros(len(xvals))
    tfs = np.zeros(len(xvals))
    sorted_index = 0
    for ifile in sorted_file_indices:
        gammatavgs[sorted_index] = full_floquet[ifile]['gamma_avg']
        gammainsts[sorted_index] = full_floquet[ifile]['gamma_max']
        gammainst_stds[sorted_index] = full_floquet[ifile]['gamma_max_std']
        tfs[sorted_index] = abs(2*np.pi*full_floquet[ifile]['shat']/full_floquet[ifile]['gexb'])
        sorted_index += 1
        
    pdflist = [] 
    tmp_pdf_id=0
    
    plt.figure(figsize=(12,8))
    ax = plt.gca()
    ax2 = ax.twinx()
    
    gplot.plot_1d(xvals,gammatavgs, xlab=x,ylab=r'$\langle\gamma\rangle_{t} a_{GS2}/v_{th,ref}$', grid = "both", marker='s', axes=ax, color='blue')
    gplot.plot_1d(xvals,gammatavgs*tfs, xlab=x,ylab=r'$T_{\rm{floq}}\langle\gamma\rangle_{t} a_{GS2}/v_{th,ref}$', grid = "both", marker='s', axes=ax2, color='red')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    gplot.plot_1d(xvals,gammainsts, xlab=x,title=r'$\gamma_{\rm{max}} a_{GS2}/v_{th,ref}$', grid = "both",errors=gammainst_stds, marker='s')
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1


    merged_pdfname = str(x) + "_scan"
    gplot.merge_pdfs(pdflist, merged_pdfname, run)
   
# Function for scanning in the parameters x and y. We expect typically x to be ky, but it could be something else.
def scan2d(run,x,y):
    # Only execute if plotting
    if run.no_plot:
        return

    print("Plotting various quantities for scan of ("+str(x)+" vs "+str(y)+")...")
    Nfile = len(run.fnames)

    # Init arrays of data used in scan.
    full_floquet = [dict() for ifile in range(Nfile)]
    
    # Get floquet data from .dat file.
    for ifile in range(Nfile):
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.floquet.dat'
        with open(datfile_name,'rb') as datfile:
            full_floquet[ifile] = pickle.load(datfile)
   
    xdat,ydat = np.zeros(Nfile),np.zeros(Nfile)
    
    for ifile in range(Nfile):
        lastcharx = x[-1]
        lastchary = y[-1]

        xstem = x[:len(x)-1]
        ystem = y[:len(y)-1]
        if xstem in ['dens','temp','fprim','tprim','vnewk']:
            if lastcharx == 'i':
                specindex = 1
            if lastcharx == 'e':
                specindex = 2
            if lastcharx == 'x':
                specindex = 3
            xdat[ifile] = full_floquet[ifile]['specparams'][specindex-1][xstem]
        elif x in ['dens','temp','fprim','tprim','vnewk']:
            xdat[ifile] = full_floquet[ifile]['specparams'][0][x]
        elif x == 'ky':
            # We assume that there is only a single ky in the list
            xdat[ifile] = round(full_floquet[ifile]['ky'][full_floquet[ifile]['iky_list'][0]],3)
        else:
            xdat[ifile] = full_floquet[ifile][x]
        if ystem in ['dens','temp','fprim','tprim','vnewk']:
            if lastchary == 'i':
                specindex = 1
            if lastchary == 'e':
                specindex = 2
            if lastchary == 'x':
                specindex = 3
            ydat[ifile] = full_floquet[ifile]['specparams'][specindex-1][xstem]
        elif y in ['dens','temp','fprim','tprim','vnewk']:
            ydat[ifile] = full_floquet[ifile]['specparams'][0][y]
        elif y == 'ky':
            # We assume that there is only a single ky in the list
            ydat[ifile] = full_floquet[ifile]['ky'][full_floquet[ifile]['iky_list'][0]]
        else:
            ydat[ifile] = full_floquet[ifile][y]

    xvals = sorted(list(set(xdat)))
    yvals = sorted(list(set(ydat)))
    print(str(x) + " values: " + str(xvals))
    print(str(y) + " values: " + str(yvals))
    if len(xvals) * len(yvals) != Nfile:
        quit("Incorrect number of files added to populate the scan - exiting")
    gammatavgs = np.zeros((len(xvals), len(yvals)))
    tfs = np.zeros((len(xvals), len(yvals)))
    gammainsts = np.zeros((len(xvals), len(yvals)))
    gammainst_stds = np.zeros((len(xvals), len(yvals)))
    for ifile in range(Nfile):
        # Limits search to ITG.
        for ix in range(len(xvals)):
            for iy in range(len(yvals)):
                if xdat[ifile] == xvals[ix] and ydat[ifile] == yvals[iy]:
                    gammatavgs[ix,iy] = full_floquet[ifile]['gamma_avg']
                    gammainsts[ix,iy] = full_floquet[ifile]['gamma_max']
                    gammainst_stds[ix,iy] = full_floquet[ifile]['gamma_max_std']
                    tfs[ix,iy] = abs(2*np.pi*full_floquet[ifile]['shat']/full_floquet[ifile]['gexb'])
    
    pdflist = [] 
    tmp_pdf_id=0
    fig = plt.figure(figsize=(12,8))
    ax1=plt.gca()
    ax2 = ax1.twinx()

    z = [gammatavgs[:,iy]*tfs[:,iy] for iy in range(len(yvals))]
    ylab = r'$T_{\rm{floq}}\langle\gamma\rangle_{{T_{\rm{floq}}}} a_{GS2}/v_{th,ref}$'
    gplot.plot_multi_1d(xvals, z, scanlabels[x], axes=ax2, legendtitle=y, ylab=ylab , grid='both', linestyle = '--')

    z = [gammatavgs[:,iy] for iy in range(len(yvals))]
    labs = ["{:.2f}".format(yvals[iy]) for iy in range(len(yvals))]
    ylab = r'$\langle\gamma\rangle_{{T_{\rm{floq}}}} a_{GS2}/v_{th,ref}$'
    gplot.plot_multi_1d(xvals, z, scanlabels[x], axes=ax1, labels=labs, legendtitle=y, ylab=ylab , grid='both')

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    fig = plt.figure(figsize=(12,8))
    axes=plt.gca()
    gplot.plot_multi_1d(xvals, [gammainsts[:,iy] for iy in range(len(yvals))], scanlabels[x], axes=axes, labels=["{:.2f}".format(yvals[iy]) for iy in range(len(yvals))], legendtitle=y, ylab = r'$\gamma_{\rm{max}} a_{GS2}/v_{th,ref}$', grid='both',errors=[gammainst_stds[:,iy] for iy in range(len(yvals))])
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id += 1

    merged_pdfname = str(x) + "_" + str(y) + "_scan"
    gplot.merge_pdfs(pdflist, merged_pdfname, run)
 

