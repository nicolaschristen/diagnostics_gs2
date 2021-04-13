from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
import numpy as np
import pickle
import copy as cp
import os
import math
import gs2_plotting as gplot
from plot_phi2_vs_time import plot_phi2_ky_vs_t

def my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields):

    # Compute and save to dat file
    if not run.only_plot:
        # OB 140918 ~ mygrids.ny = naky. Before, ny was defined as mygrids.ny, but I have changed it to ny specified in the input file, so we have both.
        #             (Same for x, and I replaced all further ny,nx that should have been naky and nakx, respectively)
        
        phi_t = myout['phi_t']          # t, ky, kx, theta, ri
        if phi_t is None:
            print('phi_t not present. Finishing.')
            return

        shat = myin['theta_grid_parameters']['shat']
        gds2 = myout['gds2']
        gds21 = myout['gds2']
        gds22 = myout['gds2']
        
        phi2 = np.power(phi_t[:,:,:,:,0],2) + np.power(phi_t[:,:,:,:,1],2)    # t, ky, kx, theta
        
        theta = mygrids.theta
        ntheta = mygrids.ntheta
        ky = mygrids.ky
        kx = mygrids.kx
        naky = len(ky)
        nakx = len(kx)

        def GammaFunc(y):
            if (y>700.0):
                return 1./np.sqrt(2*math.pi*y)
            else:
                return np.i0(y)*np.exp(-y)
    
        gamma = np.zeros((naky,nakx,ntheta))
        kperp2 = np.zeros((naky,nakx,ntheta))
        for iky in range(naky):
            for ikx in range(nakx):
                for itheta in range(ntheta):
                    if ky[iky] == 0:
                        kperp2[iky,ikx,itheta] = np.power(kx[ikx]/shat,2) * gds22[itheta] 
                        gamma[iky,ikx,itheta] = GammaFunc(kperp2[iky,ikx,itheta])
                    else:
                        theta0 = kx[ikx]/(shat*ky[iky])
                        kperp2[iky,ikx,itheta] = np.power(ky[iky],2) * ( gds2[itheta] + 2*theta0*gds21[itheta] + np.power(theta0,2) * gds22[itheta] ) 
                        gamma[iky,ikx,itheta] = GammaFunc(kperp2[iky,ikx,itheta])
        
       
        wzf_theta = np.zeros((mytime.ntime,ntheta))
        wtot_theta = np.zeros((mytime.ntime,ntheta))
        
        for i in range(nakx):
            for j in range(naky):
                if (j==0):
                    wzf_theta = wzf_theta + ( 1 - gamma[j,i,:] ) * phi2[:,j,i,:]
                wtot_theta = wtot_theta + ( 1 - gamma[j,i,:] ) * phi2[:,j,i,:]

        # Get theta-average of energies.
        wzf = np.trapz(wzf_theta, x=theta, axis=1)/(2*math.pi)
        wtot = np.trapz(wtot_theta, x=theta, axis=1)/(2*math.pi)

        wfrac = wzf/wtot        
        
        # wfrac_tavg is an array of averages where we use different time windows to average over.
        ntwins = 99
        wfrac_tavg = np.zeros(ntwins)
        twins = np.linspace(0.01,0.99,ntwins)
        for itwin in range(ntwins):
            wfrac_tavg[itwin] = mytime.timeavg(wfrac, custom_twin=twins[itwin])

        mean_wfrac_tavg = np.mean(wfrac_tavg)
        stddev_on_mean = np.sqrt(np.sum(np.power(wfrac_tavg - mean_wfrac_tavg,2))/(ntwins-1))
        stderr_on_mean = stddev_on_mean/np.sqrt(ntwins)
        
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.wzonal.dat'
        mydict = {'wzf':wzf, 'wtot':wtot, 'wfrac':wfrac, 'wfrac_tavg':wfrac_tavg, 'twins':twins, 'mean_wfrac_tavg':mean_wfrac_tavg, 'stderr_wfrac_tavg':stderr_on_mean, 'stddev_wfrac_tavg':stddev_on_mean}
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mydict,datfile)

        # Save fields obj
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fields.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(myfields,datfile)

        # Save time obj
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.time.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mytime,datfile)


        # Save grid obj
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.grids.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mygrids,datfile)

    # Read from dat files
    else:
        
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.wzonal.dat'
        with open(datfile_name,'rb') as datfile:
            mydict = pickle.load(datfile)

        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.fields.dat'
        with open(datfile_name,'rb') as datfile:
            myfields = pickle.load(datfile)
        
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            mytime = pickle.load(datfile)

        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.grids.dat'
        with open(datfile_name,'rb') as datfile:
            mygrids = pickle.load(datfile)
    

    if not run.no_plot:          # plot fluxes for this single file
            
        plot(ifile,run,mytime,mydict,myfields,mygrids)


def plot(ifile,run,mytime,mydict,myfields,mygrids):
   
    tmp_pdf_id=1
    pdflist=[]

    twins = mydict['twins']
    wfrac_tavg = mydict['wfrac_tavg']

    mean_wfrac_tavg = mydict['mean_wfrac_tavg']
    stddev_wfrac_tavg = mydict['stddev_wfrac_tavg']

    wfrac = mydict['wzf']/mydict['wtot']

    wfrac_halftavg = wfrac_tavg[ np.argmin(np.absolute(twins-0.5)) ]
    stddev_wfrac_halftavg = np.sqrt(np.sum(np.power(wfrac - wfrac_halftavg,2))/(len(wfrac)-1))

    twin = 0.5
    half_wfrac = wfrac[int(twin*len(wfrac)):]
    mean_wfrac = np.mean(half_wfrac)
    stddev_wfrac = np.sqrt(np.sum(np.power(half_wfrac - mean_wfrac,2))/(len(half_wfrac)-1))
    stderr_wfrac = stddev_wfrac/np.sqrt(len(half_wfrac))

    print('Time-averaged zonal flow fractional contribution is {:2.2f}'.format(mytime.timeavg(wfrac)))

    title = r'$W_{ZF}/W_{\rm{tot}}$'
    gplot.plot_1d(mytime.time,wfrac,r'$t$',title)
    plt.axhline(y=mean_wfrac_tavg)
    plt.axhline(y=mean_wfrac_tavg+stddev_wfrac_halftavg)
    plt.axhline(y=mean_wfrac_tavg-stddev_wfrac_halftavg)
    plt.grid(True)
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1
    
    title = r'$<W_{ZF}/W_{\rm{tot}}>_t$'
    gplot.plot_1d(twins,wfrac_tavg,r'Time window',title)
    plt.axhline(y=mean_wfrac_tavg)
    plt.axhline(y=mean_wfrac_tavg+stddev_wfrac_tavg)
    plt.axhline(y=mean_wfrac_tavg-stddev_wfrac_tavg)
    plt.grid(True)
    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    gplot.save_plot(tmp_pdfname, run, ifile)
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1
          
    merged_pdfname = 'wzonal'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)


    print('complete')
    

