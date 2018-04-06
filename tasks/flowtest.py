import math
import numpy as np
import matplotlib.pyplot as plt

import gs2_plotting as gplot

myred = [183./255, 53./255, 53./255]
myblue = [53./255, 118./255, 183./255]

def store(myin, myout, task_space):

        task_space.t = myout['t']
        task_space.dt = myin['knobs']['delt']
        task_space.g_exb = myin['dist_fn_knobs']['g_exb']
        task_space.shat = myin['theta_grid_parameters']['shat']
        task_space.jtwist = myin['kt_grids_box_parameters']['jtwist']

        task_space.kx = myout['kx']
        task_space.dky = 1./myin['kt_grids_box_parameters']['y0']
        task_space.dkx = 2.*math.pi*task_space.shat*task_space.dky/task_space.jtwist
        task_space.N = task_space.dkx/(task_space.g_exb*task_space.dt*task_space.dky)
        nakx = 2*(myin['kt_grids_box_parameters']['nx']-1)/3 + 1
        ikx_max = int(round((nakx-1)/2))
        ikx_min = ikx_max+1

        ikx = 0
        #ikx = int(round(-2./task_space.dkx)) # -dkx
        #task_space.phi2 = myout['phi2']
        task_space.phi2 = myout['phi2_by_mode'][:,1,ikx]
        #task_space.phi2 = myout['phi2_by_ky'][:,1]
        #print(task_space.phi2.shape)

def plot(run, full_space):

    dt = np.zeros(len(run.fnames)//2)
    dkx = np.zeros(len(run.fnames)//2)
    phi2 = np.zeros(len(run.fnames)//2)
    
    dt_new = np.zeros(len(run.fnames)//2)
    dkx_new = np.zeros(len(run.fnames)//2)
    phi2_new = np.zeros(len(run.fnames)//2)
    
    ilast = len(run.fnames)
    imid = len(run.fnames)//2

    ## files with old algorithm
    for ifile in range(imid):
        dt[ifile] = full_space[ifile]['flowtest'].dt
        dkx[ifile] = full_space[ifile]['flowtest'].dkx
    for ifile in range(imid):
        #it = int(round(10.*dt[0]/dt[ifile])) - 1
        it = -1
        phi2[ifile] = full_space[ifile]['flowtest'].phi2[it]
    
    ## files with new algorithm
    for ifile in range(imid, ilast):
        dt_new[ifile-imid] = full_space[ifile]['flowtest'].dt
        dkx_new[ifile-imid] = full_space[ifile]['flowtest'].dkx
    for ifile in range(imid, ilast):
        #it = int(round(10.*dt[0]/dt_new[ifile-imid])) - 1
        it = -1
        phi2_new[ifile-imid] = full_space[ifile]['flowtest'].phi2[it]

    idxsort = np.argsort(dt)
    dt = dt[idxsort]
    dt_new = dt_new[idxsort]
    phi2 = phi2[idxsort]
    phi2_new = phi2_new[idxsort]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=25)
    
    plt.figure(figsize=(12,8))
    
    #plt.title('$\\Delta k_x = \\gamma_E\\times\\Delta k_y\\times\\Delta t$')
    #plt.title('Fixed $\\Delta k_x$')
    plt.title('Fixed $\\Delta t$')
    
    #plt.plot(dt, phi2, marker='o', markersize=12, markerfacecolor='none', markeredgecolor=myblue, color=myblue, linewidth=3.0)
    #plt.plot(dt_new, phi2_new, marker='s', markersize=8, color=myred, linewidth=3.0)
    plt.plot(dkx, phi2, marker='o', markersize=12, markerfacecolor='none', markeredgecolor=myblue, color=myblue, linewidth=3.0)
    plt.plot(dkx_new, phi2_new, marker='s', markersize=8, color=myred, linewidth=3.0)

    #plt.xlabel('$\\Delta t$')
    plt.xlabel('$\\Delta k_x$')

    plt.ylabel('$\\frac{1}{2\\pi}\\int d\\theta\\vert \\phi \\vert ^2(k_x=0)$')
    plt.legend(['old algo.', 'new algo.'], loc='lower right')
    plt.gca().set_ylim(bottom = 0.0, top = 1.3e-5)
    plt.gca().get_yaxis().get_major_formatter().set_powerlimits((0.01,100.))
    plt.grid(True)

    #pdfname = 'get_converged'
    #pdfname = 'dt_scan'
    pdfname = 'dkx_scan_kx0'

    gplot.save_plot(pdfname, run)
