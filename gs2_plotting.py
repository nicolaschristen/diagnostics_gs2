import matplotlib
matplotlib.use('PDF')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PyPDF2 import PdfFileMerger, PdfFileReader
import os
import scipy.interpolate as scinterp
from math import pi

myred = [183./255, 53./255, 53./255]
myblue = [53./255, 118./255, 183./255]
oxblue = [0.,33./255,71./255]
oxbluel = [68./255,104./255,125./255]
oxbluell = [72./255,145./255,220./255]
midLoRed = [255./255,178./255,172./255]
midHiRed = [179./255,26./255,0./255]
darkRed = [102./255,26./255,0./255]
midLoBlue = [179./255,217./255,255./255]
midHiBlue = [0./255,115./255,153./255]
darkBlue = [0./255,0./255,77./255]
myredstd = [217./255, 83./255, 25./255]
mybluestd = [0./255, 114./255, 189./255]
myyellow = [237./255,177./255,32./255]
mygreen = [51./255,153./255,51./255]

def RdBu_centered(minVal, maxVal, center=0.0):

    if minVal<center and maxVal>center:
        bluePart = abs(center-minVal)/(maxVal-minVal)
    elif minVal>center:
        bluePart = 0.0
    elif maxVal<center:
        bluePart = 1.0
    c = mcolors.ColorConverter().to_rgb
    seq = [darkBlue, midHiBlue, bluePart/3.0, \
            midHiBlue, midLoBlue, 2.0*bluePart/3.0, \
            midLoBlue, c('white'), bluePart, \
            c('white'), midLoRed, (1.0-bluePart)/3.0+bluePart, \
            midLoRed, midHiRed, 2.0*(1.0-bluePart)/3.0+bluePart, \
            midHiRed, darkRed]
    #print(bluePart)
    #seq = [c('blue'), c('white'), bluePart, c('white'), c('red')]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def save_plot(pdfname, run, ifile = None):

    if ifile is not None:
        filename = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
    else:
        filename = run.out_dir + pdfname + '.pdf'
    
    plt.savefig(filename)
    plt.cla()
    plt.clf()

def merge_pdfs(in_namelist, out_name, run, ifile = None):

    # read all tmp pdfs to be merged
    merger = PdfFileMerger()
    for pdfname in in_namelist:
        if ifile is not None:
            file_name = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
        else:
            file_name = run.out_dir + pdfname + '.pdf'
        with open(file_name, 'rb') as pdffile:
            merger.append(PdfFileReader(pdffile))

    # write and save output pdf
    if ifile is not None:
        out_name = run.out_dir + out_name + '_' + run.fnames[ifile] + '.pdf'
    else:
        out_name = run.out_dir + out_name + '.pdf'
    merger.write(out_name)

    # remove tmp pdfs
    for pdfname in in_namelist:
        if ifile is not None:
            file_name = run.out_dir + pdfname + '_' + run.fnames[ifile] + '.pdf'
        else:
            file_name = run.out_dir + pdfname + '.pdf'
        os.system('rm -f '+file_name)

    plt.cla()
    plt.clf()

def set_plot_defaults():

    # setup some plot defaults
    plt.rc('text', usetex=False) # False for ARCHER, True for MARCONI
    plt.rc('font', family='serif')
    plt.rc('font', size=30)
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'legend.fontsize': 20, 'legend.handlelength': 4})
    rcParams.update({'legend.frameon': False})
    #rcParams.update({'animation.ffmpeg_path':'/marconi/home/userexternal/nchriste/codes/ffmpeg'}) # for HPC use only



def nearNeighb_interp_1d(x,y,xout):

    if isinstance(x,(list)):
        n = len(x)
    elif isinstance(x,(np.ndarray)):
        n = x.size
    if isinstance(xout,(list)):
        nout = len(xout)
        yout = [0*i for i in range(nout)]
    elif isinstance(xout,(np.ndarray)):
        nout = xout.size
        yout = np.zeros(nout)

    i = 0
    iout = 0
    for i in range(n-1):
        while xout[iout]-x[i] < (x[i+1]-x[i])/2.0:
            yout[iout] = y[i]
            iout += 1
    yout[iout:] = y[-1]

    return yout



def plot_1d(x,y,xlab,title='',ylab=''):

    fig = plt.figure(figsize=(12,8))
    plt.plot(x,y)
    plt.xlabel(xlab)
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    return fig

def plot_2d(z,xin,yin,zmin,zmax,xlab='',ylab='',title='',cmp='RdBu',use_logcolor=False,x_is_2pi=False, z_ticks=None, z_ticks_labels=None):

    fig = plt.figure(figsize=(12,8))
    x,y = np.meshgrid(xin,yin)
    dx = xin[1]-xin[0]
    dy = yin[1]-yin[0]

    # Centered blue->red color map
    if cmp=='RdBu_c':
        cmp = RdBu_centered(zmin, zmax)
    elif cmp == 'RdBu_c_one':
        cmp = RdBu_centered(zmin, zmax, center=1.0)

    if use_logcolor:
        color_norm = mcolors.LogNorm(zmin,zmax)
    else:
        color_norm = mcolors.Normalize()

    cax = plt.imshow(z, cmap=cmp, vmin=zmin, vmax=zmax,
               extent=[x.min(),x.max(),y.min()-dy/2,y.max()+dy/2],
               interpolation='nearest', origin='lower', aspect='auto',
               norm=color_norm)
    plt.axis([x.min(), x.max(), y.min()-dy/2, y.max()+dy/2])
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    if x_is_2pi:
        plt.xticks([-pi,-pi/2,0,pi/2,pi],['$-\\pi$','$-\\pi/2$','$0$','$\\pi/2$','$\\pi$'],
                fontsize=28)

    if z_ticks is None:
        z_ticks = [zmin+(zmax-zmin)*f for f in [0,0.25,0.5,0.75,1.0]]
    if z_ticks_labels is None:
        if abs(zmax-zmin) > 0.01:
            z_ticks_labels = [str(round(iz,3)) for iz in z_ticks]
        else:
            z_ticks_labels = ['{:.2E}'.format(iz) for iz in z_ticks]
    cbar = plt.colorbar(cax, ticks=z_ticks)
    cbar.ax.set_yticklabels(z_ticks_labels)
    cbar.ax.tick_params(labelsize=28)
    plt.xlabel(xlab, fontsize=32)
    plt.ylabel(ylab, fontsize=32)
    plt.title(title, fontsize=32)
    return fig

# Input:
# x = x[iy][ix]
# y = y[iy]
# z = z[iy][ix]
def plot_2d_uneven_xgrid(x, y, z, xmin, xmax, cbarmin, cbarmax, xlabel, ylabel, title, x_is_twopi=True, ngrid_fine = 1001, clrmap='RdBu_c', zticks=None, zticks_labels=None):

    # Here we assume that the scan uses a fixed set of ky.
    ny = y.size

    # Finer and regular x mesh
    ntheta0_fine = 1001
    x_fine = np.linspace(xmin, xmax, ngrid_fine)
    z_fine = np.zeros((ny, ngrid_fine))

    # For each ky, interpolate to nearest neighbour in x
    for iy in range(ny):
        z_fine[iy,:] = nearNeighb_interp_1d(x[iy],z[iy],x_fine)

    plot_2d(z_fine, x_fine, y, cbarmin, cbarmax, xlabel, ylabel, title, cmp=clrmap, x_is_2pi=x_is_twopi, z_ticks=zticks, z_ticks_labels=zticks_labels)

def movie_2d(z,xin,yin,zmin,zmax,nframes,outfile,xlab='',ylab='',title='',step=1,cmp='RdBu'):

    from matplotlib import animation

    fig = plt.figure(figsize=(12,8))
    x,y = np.meshgrid(xin,yin)
    im = plt.imshow(z[0,:,:], cmap=cmp, vmin=zmin, vmax=zmax,
               extent=[x.min(),x.max(),y.min(),y.max()],
               interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    
    ims = []

    for i in range(1,nframes,step):
        im = plt.imshow(z[i,:,:], cmap=cmp, vmin=zmin, vmax=zmax,
               extent=[x.min(),x.max(),y.min(),y.max()],
               interpolation='nearest', origin='lower', aspect='auto')
        ims.append([im])

    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True)
    ani.save(outfile)

def movie_1d(x,y,xmin,xmax,ymin,ymax,nframes):

    from matplotlib import animation

    fig = plt.figure(figsize=(12,8))
    ax=plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
    line, = ax.plot([],[],lw=2)

    def init():
        line.set_data([],[])
        return line,

    def animate(i):
        line.set_data(x,y[i,:])
        return line,

    anim=animation.FuncAnimation(fig, animate, init_func=init, 
                                 frames=nframes, interval=200)

    return anim

def str_tt0(theta0):

    n = int(round(theta0/(2.0*pi)))

    txt = '{: .2f}'.format(theta0 - 2*pi*n)

    if n > 0:
        txt += '$+' + str(2*n) + '\\pi$'
    elif n < 0:
        txt += '$-' + str(abs(2*n)) + '\\pi$'

    return txt

def str_t(time):

    return '${:.2E}$'.format(time)

def str_ky(ky):

    return '{: .2f}'.format(ky)

def legend_matlab(my_legend=None):

    if my_legend:
        legend = plt.legend(my_legend, frameon=True, fancybox=False)
    else:
        legend = plt.legend(frameon=True, fancybox=False)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_linewidth(0.5)
    frame.set_alpha(1)
