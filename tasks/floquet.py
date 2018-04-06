from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
    
    # select chain
    my_iky = 1
    my_dmid = 0

    # Select time for plot of phi vs ballooning angle
    my_it = 1

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
        
        my_vars = process_and_save_to_dat(ifile, run, myin, myout, my_dmid, my_iky)
        
    store_for_task_scan(my_vars, task_space)

    if not run.no_plot:

        plot_task_single(ifile, run, my_vars, my_it, my_iky, my_dmid, make_movies)


    print('\n... single task completed.')


def process_and_save_to_dat(ifile, run, myin, myout, my_dmid, my_iky):

    t = myout['t']
    delt = myin['knobs']['delt']
    nt = t.size

    theta = myout['theta']
    ntheta = theta.size
    theta0 = myout['theta0']
    
    g_exb = myin['dist_fn_knobs']['g_exb']
    shat = myin['theta_grid_parameters']['shat']
    jtwist = int(myin['kt_grids_box_parameters']['jtwist'])
    
    # number of t-steps in Floquet period
    Tf = 2*pi*shat/g_exb
    print('Floquet period : ' + str(Tf))
    Nf = int(round(Tf/delt))
    print('Number of t-steps in Floquet period : ' + str(Nf))

    kx_gs2 = myout['kx']
    ky = myout['ky']
    dky = 1./myin['kt_grids_box_parameters']['y0']
    dkx = 2.*pi*shat*dky/jtwist
    nakx = kx_gs2.size
    naky = ky.size
    ikx_max = int(round((nakx-1)/2))
    ikx_min = ikx_max+1
    
    # number of t-steps before ExB re-map
    N = int(round(dkx/(g_exb*delt*dky)))
    print('Number of t-steps before ExB re-map : ' + str(N))

    phi2_gs2 = myout['phi2_by_mode'][:,:,:]

    phi_t_present = myout['phi_t_present']
    if phi_t_present:
        phi2_bytheta_gs2 = np.sum(np.power(myout['phi_t'],2), axis=4)
   
    # sorting kx_gs2 to get monotonic kx_star
    kx_star = np.concatenate((kx_gs2[ikx_min:],kx_gs2[:ikx_min]))
    phi2 = np.concatenate((phi2_gs2[:,:,ikx_min:], phi2_gs2[:,:,:ikx_min]), axis=2)
    if phi_t_present:
        phi2_bytheta = np.concatenate((phi2_bytheta_gs2[:,:,ikx_min:,:], phi2_bytheta_gs2[:,:,:ikx_min,:]), axis=2)

    # get kx from kx_star
    kx = np.zeros((nt,naky,nakx))
    for it in range(nt):
        for iky in range(naky):
            ikx_shift = int(round(g_exb*ky[iky]*delt*it/dkx))
            for ikx in range(nakx):
                kx[it,iky,ikx] = kx_star[ikx] + ikx_shift*dkx
 
    # index of kx=0
    ikx0 = (nakx-1)//2

    
    #
    # Compute growthrate at midplane for every kx + averaged over a Floquet period
    #

    # non-averaged version
    ikx_shift_old = 0
    gamma_mid = np.zeros((nt,nakx))
    for it in range(1,nt):
        ikx_shift = int(round(g_exb*ky[my_iky]*delt*it/dkx))
        shifted = ikx_shift - ikx_shift_old
        for ikx in range(nakx):
            if ikx + shifted >= 0 and ikx + shifted < nakx:
                if phi2_bytheta[it-1,my_iky,ikx+shifted,(ntheta-1)//2]==0:
                    print('phi is zero')
                    print('it='+str(it))
                    print('ikx='+str(ikx+shifted))
                gamma_mid[it,ikx] = gamma_mid[it,ikx] + \
                        1./(2.*delt)*np.log(phi2_bytheta[it,my_iky,ikx,(ntheta-1)//2]/phi2_bytheta[it-1,my_iky,ikx+shifted,(ntheta-1)//2])
            else:
                gamma_mid[it,ikx] = np.nan
        ikx_shift_old = ikx_shift

    # averaged version

    gamma_mid_avg = np.zeros(((nt-1)//Nf,nakx))
    ikx_shift_old = 0
    it = 1
    for ifloq in range(((nt-1)//Nf)):
        while (it <= (ifloq+1)*Nf):
            ikx_shift = int(round(g_exb*ky[my_iky]*delt*(it-ifloq*Nf)/dkx))
            for ikx in range(nakx):
                if ((ikx-ikx_shift) >= 0 and (ikx-ikx_shift) < nakx):
                    if phi2_bytheta[it-1,my_iky,ikx-ikx_shift_old,(ntheta-1)//2]==0:
                        print('phi is zero')
                    gamma_mid_avg[ifloq,ikx] = gamma_mid_avg[ifloq,ikx] + \
                            1./(2.*delt)*np.log(phi2_bytheta[it,my_iky,ikx-ikx_shift,(ntheta-1)//2] / phi2_bytheta[it-1,my_iky,ikx-ikx_shift_old,(ntheta-1)//2])
                else:
                    gamma_mid_avg[ifloq,ikx] = np.nan
            ikx_shift_old = ikx_shift
            it = it+1
        ikx_shift_old = 0
        gamma_mid_avg[ifloq,:] = gamma_mid_avg[ifloq,:]/Nf


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
    
    sum_phi2_chain = []
    bloonang_chain = []
    phi2bloon_chain = []
    gamma_chain = []

    ikx_shift_old = 0

    for it in range(nt):

        ikx_members = []
        sum_phi2_now = 0
        bloonang_now = []
        phi2bloon_now = []
        gamma_now = []

        # BLACK MAGIC LINE :
        # if the position of delt and it are swapped in the following multiplication,
        # the resulting ikx_shift can be different ! (e.g. it=297 for ~/gs2/flowtest/dkx_scan/dkx_2.in)
        ikx_shift = int(round(g_exb*ky[my_iky]*delt*it/dkx))
    
        # fill in part with kx<=0
        # ikx such that kx(kx_star[it,ikx]) = dmid*dkx
        ikx = ikx0 - ikx_shift + my_dmid
        while (ikx >= nakx):
            ikx = ikx-jtwist*my_iky
        while (ikx >= 0):
            ikx_members.append(ikx)
            sum_phi2_now = sum_phi2_now + phi2[it,my_iky,ikx]                    
            ikx = ikx-jtwist*my_iky

        # fill in part with kx>0
        ikx = ikx0 - ikx_shift + my_dmid + jtwist*my_iky
        while (ikx < 0):
            ikx = ikx+jtwist*my_iky
        while (ikx < nakx):
            ikx_members.append(ikx)
            sum_phi2_now = sum_phi2_now + phi2[it,my_iky,ikx]
            ikx = ikx+jtwist*my_iky

        # sort ikx of chain members in ascending order
        ikx_members = np.sort(ikx_members)

        # compute ballooning angle and construct associated phi2
        if phi_t_present:
            for imember in range(len(ikx_members)-1,-1,-1):
                for itheta in range(ntheta):
                    bloonang = theta[itheta] - kx[it,my_iky,ikx_members[imember]]/(shat*ky[my_iky])
                    bloonang_now.append(bloonang)
                    phi2bloon_now.append(phi2_bytheta[it,my_iky,ikx_members[imember],itheta])

                    # compute growthrate
                    shifted = ikx_shift - ikx_shift_old
                    if it > 0:
                        if ikx_members[imember] + shifted >= 0 and ikx_members[imember] + shifted < nakx:
                            gamma = 1./(2.*delt)*np.log(phi2_bytheta[it,my_iky,ikx_members[imember],itheta]/phi2_bytheta[it-1,my_iky,ikx_members[imember]+shifted,itheta])
                        else:
                            gamma = np.nan
                    else:
                        gamma = 0.
                    gamma_now.append(gamma)

        sum_phi2_chain.append(sum_phi2_now)
        bloonang_chain.append(bloonang_now)
        phi2bloon_chain.append(phi2bloon_now)
        gamma_chain.append(gamma_now)

        ikx_shift_old = ikx_shift
    
    # Saving variables to mat-file
    my_vars = {}
    my_vars['Nf'] = Nf
    my_vars['t'] = t
    my_vars['delt'] = delt
    my_vars['kx'] = kx
    my_vars['kx_star'] = kx_star
    my_vars['dkx'] = dkx
    my_vars['bloonang_chain'] = bloonang_chain
    my_vars['phi2'] = phi2
    my_vars['phi2bloon_chain'] = phi2bloon_chain
    my_vars['sum_phi2_chain'] = sum_phi2_chain
    my_vars['gamma_mid'] = gamma_mid
    my_vars['gamma_mid_avg'] = gamma_mid_avg
    my_vars['gamma_chain'] = gamma_chain
    my_vars['phi_t_present'] = phi_t_present

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
    
def plot_task_single(ifile, run, my_vars, my_it, my_iky, my_dmid, make_movies):
        
    Nf = my_vars['Nf']
    t = my_vars['t']
    delt = my_vars['delt']
    kx = my_vars['kx']
    kx_star = my_vars['kx_star']
    dkx = my_vars['dkx']
    bloonang_chain = my_vars['bloonang_chain']
    phi2 = my_vars['phi2']
    phi2bloon_chain = my_vars['phi2bloon_chain']
    sum_phi2_chain = my_vars['sum_phi2_chain']
    gamma_mid = my_vars['gamma_mid']
    gamma_mid_avg = my_vars['gamma_mid_avg']
    gamma_chain = my_vars['gamma_chain']
    phi_t_present = my_vars['phi_t_present']
    
    Tf = Nf*delt
    nt = t.size

    plt.figure(figsize=(12,8))
    
    # plot sum of phi2 along chain vs time
    plt.xlabel('$t$')
    plt.ylabel('$\\ln \\left(\\sum_{K_x}\\vert \\langle \\phi \\rangle_\\theta \\vert ^2\\right)$')
    plt.title('Sum along a single ballooning mode')
    plt.grid(True)
    plt.plot(t, np.log(sum_phi2_chain), color=gplots.myblue, linewidth=3.0) 
    pdfname = 'floquet_vs_t'
    pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid) + '.pdf'
    plt.savefig(pdfname)
    
    plt.clf()
    plt.cla()

    # plot phi2 of chosen chain vs ballooning angle at chosen time
    if (phi_t_present):
        
        plt.xlabel('$\\theta -\\theta_0$')
        plt.ylabel('$\\vert \\phi \\vert ^2$')
        plt.title('$t=$ '+str(t[my_it]))
        plt.grid(True)
        plt.gca().set_xlim(np.min(bloonang_chain[my_it]),np.max(bloonang_chain[my_it]))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
        plt.plot(bloonang_chain[my_it], phi2bloon_chain[my_it], marker='o', \
                markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)

        pdfname = 'balloon_it_' + str(my_it)
        pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid) + '.pdf'
        plt.savefig(pdfname)
        
        plt.clf()
        plt.cla()

    # make movie of phi2 vs ballooning angle over time
    if (make_movies and phi_t_present):
        
        moviename = run.out_dir + 'phi_bloon_' + run.fnames[ifile] + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid) + '.mp4'
        images = []

        # find global min and max of ballooning angle
        bloonang_min = 0.
        bloonang_max = 0.
        for it in range(nt):
            if np.min(bloonang_chain[it]) < bloonang_min:
                bloonang_min = np.min(bloonang_chain[it])
            if np.max(bloonang_chain[it]) > bloonang_max:
                bloonang_max = np.max(bloonang_chain[it])
       
        print("\ncreating movie of phi vs ballooning angle ...")
        for it in range(nt):
            
            sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1)))
       
            plt.xlabel('$\\theta -\\theta_0$')
            plt.ylabel('$\\vert \\phi \\vert ^2$')
            plt.title('$t=$ '+str(t[it]))
            plt.grid(True)
            plt.gca().set_xlim(bloonang_min,bloonang_max)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            plt.plot(bloonang_chain[it], phi2bloon_chain[it], marker='o', \
                    markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)

            pngname = run.out_dir + 'tmp_image.png'
            plt.savefig(pngname)
            
            images.append(imageio.imread(pngname))
            os.system('rm -rf ' + pngname)

            plt.clf()
            plt.cla()

            sys.stdout.flush()
        
        imageio.mimsave(moviename, images, format='FFMPEG')
        print("\n... movie completed.")
    
    # plot instantaneous growthrate at mid-plane vs kx
    pdflist = []
    for ifloq in range((nt-1)//Nf):
        plt.xlabel('$k_x$')
        plt.ylabel('$\\gamma$')
        plt.title('Growthrate at $\\theta = 0$ and $t='+str(t[ifloq*Nf])+'$')
        plt.grid(True)
        plt.plot(kx[ifloq*Nf,my_iky,:], gamma_mid[ifloq*Nf,:], color=gplots.myblue)
        
        pdfname = 'tmp_' + str(ifloq)
        gplots.save_plot(pdfname, run, ifile)
        
        plt.clf()
        plt.cla()

        pdflist.append(pdfname)
    outname = 'growth_mid'
    outname = outname + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid)
    gplots.merge_pdfs(pdflist,outname,run,ifile)

    # plot Floquet-averaged growthrate at mid-plane vs kx
    
    pdflist = []

    for ifloq in range((nt-1)//Nf):
        plt.xlabel('$k_x$')
        plt.ylabel('$\\langle\\gamma\\rangle_{T_F}$')
        plt.title('Growthrate at $\\theta = 0$ averaged over $'+"{:0.1f}".format(t[ifloq*Nf])+'\\leq t\\leq'+ "{:0.1f}".format(t[(ifloq+1)*Nf-1]) +'$')
        plt.grid(True)
        plt.plot(kx[ifloq*Nf,my_iky,:], gamma_mid_avg[ifloq,:], color=gplots.myblue)
        
        pdfname = 'tmp_' + str(ifloq)
        gplots.save_plot(pdfname, run, ifile)
        
        plt.clf()
        plt.cla()

        pdflist.append(pdfname)

    outname = 'growth_mid_avg'
    outname = outname + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid)
    gplots.merge_pdfs(pdflist,outname,run,ifile)
    
    # make movie of growthrate at mid-plane vs kx over time
    if (make_movies and phi_t_present):
        
        moviename = run.out_dir + 'growth_mid_' + run.fnames[ifile] + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid) + '.mp4'
        images = []

        # find global min and max of kx
        kx_min = 0.
        kx_max = 0.
        for it in range(nt):
            for ikx in range(nakx):
                if np.min(kx[it,my_iky,ikx]) < kx_min:
                    kx_min = np.min(kx[it,my_iky,ikx])
                if np.max(kx[it,my_iky,ikx]) > kx_max:
                    kx_max = np.max(kx[it,my_iky,ikx])
       
        print("\ncreating movie of growthrate at mid-plane vs kx ...")
        for it in range(nt):
            
            sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1)))
       
            plt.xlabel('$k_x$')
            plt.ylabel('$\\gamma (\\theta=0)$')
            plt.title('$t=$ '+str(t[it]))
            plt.grid(True)
            plt.gca().set_xlim(kx_min,kx_max)
            plt.gca().set_ylim(bottom=-0.5)
            plt.gca().set_ylim(top=0.5)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            plt.plot(kx[it,my_iky,:], gamma_mid[it,:], marker='o', \
                    markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)

            pngname = run.out_dir + 'tmp_image.png'
            plt.savefig(pngname)
            
            images.append(imageio.imread(pngname))
            os.system('rm -rf ' + pngname)

            plt.clf()
            plt.cla()

            sys.stdout.flush()
        
        imageio.mimsave(moviename, images, format='FFMPEG')
        print("\n... movie completed.")

    # plot growthrate vs theta-theta0 at time t[my_it]
    plt.xlabel('$\\theta - \\theta_0$')
    plt.ylabel('$\\gamma$')
    plt.title('Growthrate at $t='+str(t[my_it])+'$')
    plt.grid(True)
    plt.plot(bloonang_chain[my_it], gamma_chain[my_it][:], color=gplots.myblue)

    pdfname = 'growth_bloon_it_'+str(my_it)
    pdfname = run.out_dir + pdfname + '_' + run.fnames[ifile] + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid) + '.pdf'
    plt.savefig(pdfname)
    
    plt.clf()
    plt.cla()

    # make movie of growthrate vs ballooning angle over time
    if (make_movies and phi_t_present):
        
        moviename = run.out_dir + 'growth_bloon_' + run.fnames[ifile] + '_iky_' + str(my_iky) + '_dmid_' + str(my_dmid) + '.mp4'
        images = []

        # find global min and max of ballooning angle
        bloonang_min = 0.
        bloonang_max = 0.
        for it in range(nt):
            if np.min(bloonang_chain[it]) < bloonang_min:
                bloonang_min = np.min(bloonang_chain[it])
            if np.max(bloonang_chain[it]) > bloonang_max:
                bloonang_max = np.max(bloonang_chain[it])
       
        print("\ncreating movie of growthrate vs ballooning angle ...")
        for it in range(nt):
            
            sys.stdout.write("\r{0}".format("\tFrame : "+str(it)+"/"+str(nt-1)))
       
            plt.xlabel('$\\theta -\\theta_0$')
            plt.ylabel('$\\gamma$')
            plt.title('$t=$ '+str(t[it]))
            plt.grid(True)
            plt.gca().set_xlim(bloonang_min,bloonang_max)
            plt.gca().set_ylim(bottom=-0.5)
            plt.gca().set_ylim(top=0.5)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            plt.plot(bloonang_chain[it], gamma_chain[it], marker='o', \
                    markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)

            pngname = run.out_dir + 'tmp_image.png'
            plt.savefig(pngname)
            
            images.append(imageio.imread(pngname))
            os.system('rm -rf ' + pngname)

            plt.clf()
            plt.cla()

            sys.stdout.flush()
        
        imageio.mimsave(moviename, images, format='FFMPEG')
        print("\n... movie completed.")
    
    # plot phi2 vs t for each kx
    plt.xlabel('$t$')
    my_ylabel = '$\\ln \\left(\\vert \\langle \\phi \\rangle_\\theta \\vert ^2\\right)$'
    plt.ylabel(my_ylabel)
    plt.grid(True)
    my_colorlist = plt.cm.plasma(np.linspace(0,1,kx_star.size))
    my_legend = []
    for ikx in range(kx_star.size):
        plt.plot(t, np.log(phi2[:,1,ikx]), color=my_colorlist[ikx])
        my_legend.append("kx = "+str(kx_star[ikx]))
    plt.legend(my_legend)

    pdfname = 'phi2_by_kx'
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
    task_space.dkx = my_vars['dkx']
    task_space.Tf = my_vars['Nf']*my_vars['delt']
    task_space.sum_phi2_chain = my_vars['sum_phi2_chain']


################################################################################
# Part of task for scans
################################################################################

def task_scan(run, full_space):

    # Start comparing simulations at time-step it_start = N_start*Tfloquet/dt
    # ie after N_start Floquet oscillations
    # Normalise sum_phi2 by sum_phi2[it_start] for each run
    
    N_start = 5
    
    sum_phi2 = []
    t = []
    delt = np.zeros(len(run.fnames))
    dkx = np.zeros(len(run.fnames))
    slope = np.zeros(len(run.fnames))
    
    if not run.no_plot:

        for ifile in range(len(run.fnames)):
            
            Tf = full_space[ifile]['floquet'].Tf
            delt[ifile] = full_space[ifile]['floquet'].delt
            dkx[ifile] = full_space[ifile]['floquet'].dkx

            it_start = int(round(N_start*Tf/delt[ifile]))

            sum_phi2_tmp = np.zeros(len(full_space[ifile]['floquet'].sum_phi2_chain)-it_start)
            for it in range(sum_phi2_tmp.size):
                sum_phi2_tmp[it] = full_space[ifile]['floquet'].sum_phi2_chain[it_start+it]
            sum_phi2_tmp = sum_phi2_tmp/sum_phi2_tmp[0]
            sum_phi2.append(sum_phi2_tmp)
            
            t_tmp = np.zeros(len(full_space[ifile]['floquet'].t)-it_start)
            for it in range(t_tmp.size):
                t_tmp[it] = full_space[ifile]['floquet'].t[it_start+it]
            t.append(t_tmp)

            [a,dummy] = leastsq_lin(t_tmp,np.log(sum_phi2_tmp))
            slope[ifile] = a
            
        idxsort = np.argsort(delt)
        delt = delt[idxsort]
        dkx = dkx[idxsort]
        slope = slope[idxsort]
    
        pdflist = []
        plt.figure(figsize=(12,8))
        
        plt.xlabel('$t$')
        plt.ylabel('$\\ln \\left(\\sum_{K_x}\\vert \\langle\\phi\\rangle_\\theta \\vert ^2\\right)$')
        plt.title('Sum along a single ballooning mode')
        plt.grid(True)
        my_legend = []
        my_colorlist = plt.cm.plasma(np.linspace(0,1,len(run.fnames)))
        for ifile in range(len(run.fnames)):
            #my_legend.append('$\\Delta t =$'+str(full_space[ifile]['floquet'].delt))
            my_legend.append('$\\Delta k_x =$'+str(full_space[ifile]['floquet'].dkx))
            plt.plot(t[ifile], np.log(sum_phi2[ifile]), color=my_colorlist[ifile], linewidth=3.0)
        plt.legend(my_legend)
        
        pdfname = 'tmp_1'
        gplots.save_plot(pdfname, run, ifile)
        pdflist.append(pdfname)
        
        plt.clf()
        plt.cla()

        #plt.xlabel('$\\Delta t$')
        plt.xlabel('$\\Delta k_x$')
        plt.ylabel('$\\langle \\gamma \\rangle_t$')
        plt.title('Time averaged growth-rate')
        plt.grid(True)
        print('Slopes : ',end='')
        print(slope)
        #plt.plot(delt, slope, marker='o', \
        #        markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)
        plt.plot(dkx, slope, marker='o', \
                markersize=12, markerfacecolor='none', markeredgecolor=gplots.myblue, linewidth=3.0)

        
        pdfname = 'tmp_2'
        gplots.save_plot(pdfname, run, ifile)
        pdflist.append(pdfname)
        
        plt.clf()
        plt.cla()

        outname = run.scan_name
        gplots.merge_pdfs(pdflist,outname,run,ifile)

        plt.close()

def leastsq_lin(x, y):
    
    # y_fit = a*x + b
    # minimising sum((y - f_fit)^2)
    N_x = x.size

    a = 1./(N_x*np.sum(np.power(x,2)) - np.sum(x)**2) * (N_x*np.sum(np.multiply(x,y)) - np.sum(x)*np.sum(y))
    
    b = 1./(N_x*np.sum(np.power(x,2)) - np.sum(x)**2) * (-1.*np.sum(x)*np.sum(np.multiply(x,y)) + np.sum(np.power(x,2))*np.sum(y))

    return [a, b]
