from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
import numpy as np
import pickle
import copy as cp
import os
import gs2_plotting as gplot
from gs2_fft import gs2fft
from plot_phi2_vs_time import plot_phi2_ky_vs_t

def form_complex(var):
    arr = var[..., 0] + 1j*var[..., 1]
    return arr


def my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields):

    # Compute and save to dat file
    if not run.only_plot:
        # OB 140918 ~ mygrids.ny = naky. Before, ny was defined as mygrids.ny, but I have changed it to ny specified in the input file, so we have both.
        #             (Same for x, and I replaced all further ny,nx that should have been naky and nakx, respectively)
        
   

        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.real.dat'
        if 'phi_t' in myout:
            print("Calculating real-space phi")
            phit = myout['phi_t']
            
            theta = mygrids.theta

            phi = form_complex(phit)
        
            #Phi in real-space
            mydim = (mytime.ntime, 2*mygrids.ny-1, mygrids.nx, mygrids.ntheta)
            phi_real = np.zeros(mydim, dtype=float)
            
            for it in range (mytime.ntime):
                print("Carrying out Fourier transform for time-step {}/{}...".format(it+1,mytime.ntime), end="\r")
                for itheta in range(mygrids.ntheta):
                    phi_real[it,:,:,itheta] = np.fft.fftshift(gs2fft(phi[it,:,:,itheta]*(2*mygrids.ny-1)*mygrids.nx, mygrids)[0])

            mydict = {'phit':phit, 'phi_real':phi_real }

        else:
            mydict = {'phit':None}

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
        
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.real.dat'
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

    # If true, plots the potential in real space over time, at a specific theta location.
    # If false, plots the time-averaged potential over all theta values.
    plot_real_over_time = False
    thetaVidDegrees = 0.
    
    if mydict['phit'] is not None:

        phit = mydict['phit'] 
        phi_real = mydict['phi_real']
        ims=[]
        # Make movie with phi2_30_real
        fig = plt.figure(figsize=(12,8))
        x=mygrids.xgrid
        y=mygrids.ygrid
        theta=mygrids.theta
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        if plot_real_over_time:

            print('Calculating real-space phi as close as possible to theta={:1.1f} degrees for movie'.format(thetaVidDegrees))
            
            itheta = np.argmin(np.absolute(theta-(np.pi/ 180. * thetaVidDegrees)))
            print('Theta is actually {:1.1f} degrees.'.format(180.*theta[itheta]/np.pi))

            plotfield = phi_real
            
            cmax = np.amax(plotfield[:,:,:,itheta])
            cmin = np.amin(plotfield[:,:,:,itheta])
            nframes = min(len(plotfield),400)
            for i in range(nframes):
                print("Frame {}/{}...".format(i+1,nframes), end="\r")
                im = gplot.plot_2d(np.transpose(plotfield[i,:,:,itheta]),x,y,cmin,cmax,r'$x$',r'$y$',title="",cmp='RdBu',interpolation='linear',anim=True)
                ims.append([im])
            print("Images generated. Now compiling into a video...")
            # ims is a list of lists, each row is a list of artists to draw in the
            # current frame; here we are just animating one artist, the image, in
            # each 
            fig.colorbar(ims[0][0])
            ani = animation.ArtistAnimation(fig, ims, interval=16, blit=True,
                                            repeat_delay=3000)

            ani.save(run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '_phi_real_theta={:1.1f}.mp4'.format(180.*theta[itheta]/np.pi))


        else:
            plotfield = phi_real[-1,:,:,:]          # Last timestep
            #plotfield = mytime.timeavg(phi_real)    # Time average
            cmax = np.amax(plotfield)
            cmin = np.amin(plotfield)

            nframes = len(theta)
            for i in range(nframes):
                print("Frame {}/{}...".format(i+1,nframes), end="\r")
                im = gplot.plot_2d(np.transpose(plotfield[:,:,i]),x,y,cmin,cmax,r'$x$',r'$y$',title="",cmp='RdBu',interpolation='linear',anim=True)
                ims.append([im])
            
            print("Images generated. Now compiling into a video...")
            # ims is a list of lists, each row is a list of artists to draw in the
            # current frame; here we are just animating one artist, the image, in
            # each 
            fig.colorbar(ims[0][0])
            ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True,
                                            repeat_delay=3000)

            ani.save(run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '_phi_real_tavg.mp4')

            

    else:


        fig = plt.figure(figsize=(12,8))
        x=mygrids.xgrid
        y=mygrids.ygrid
        
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        plotfield = myfields.phi_xyt
        
        ims=[]
        cmax = np.amax(plotfield[:,:,:])
        cmin = np.amin(plotfield[:,:,:])
        for i in range(min(len(plotfield),400)):
            print(i)
            im = gplot.plot_2d(np.transpose(plotfield[i,:,:]),x,y,cmin,cmax,r'$x$',r'$y$',title="",cmp='RdBu',interpolation='linear',anim=True)
            ims.append([im])
       
        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in
        # each 
        fig.colorbar(ims[0][0])
        ani = animation.ArtistAnimation(fig, ims, interval=16, blit=True,
                                        repeat_delay=3000)

        ani.save('phit_igomega.mp4')













    print('complete')
    

