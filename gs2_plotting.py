import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from PyPDF2 import PdfFileMerger, PdfFileReader
import os

from gs2_utils import radians

myred = [183./255, 53./255, 53./255]
myblue = [53./255, 118./255, 183./255]
oxblue = [0.,33./255,71./255]
oxbluel = [68./255,104./255,125./255]
oxbluell = [72./255,145./255,220./255]

def save_plot(pdfname, run, ifile = None):  
    if ifile is not None:
        filename = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
    else:
        filename = run.work_dir + pdfname + '.pdf'
    
    plt.savefig(filename)
    plt.cla()
    plt.clf()

def merge_pdfs(in_namelist, out_name, run, ifile = None):

    # read all tmp pdfs to be merged
    merger = PdfFileMerger()
    for pdfname in in_namelist:
        if ifile is not None:
            file_name = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
        else:
            file_name = run.work_dir + pdfname + '.pdf'
        with open(file_name, 'rb') as pdffile:
            merger.append(PdfFileReader(pdffile))

    # write and save output pdf
    if ifile is not None:
        out_name = run.work_dir + run.dirs[ifile] + run.out_dir + out_name + '_' + run.files[ifile] + '.pdf'
    else:
        out_name = run.work_dir + out_name + '.pdf'
    merger.write(out_name)

    # remove tmp pdfs
    for pdfname in in_namelist:
        if ifile is not None:
            file_name = run.work_dir + run.dirs[ifile] + run.out_dir + pdfname + '_' + run.files[ifile] + '.pdf'
        else:
            file_name = run.work_dir + pdfname + '.pdf'
        os.system('rm -f '+file_name)

    plt.cla()
    plt.clf()

def set_plot_defaults():
    import os
    # setup some plot defaults OB 200918 ~ hacky solution to prevent latex drawing on archer (cray system).
    plt.rc('text', usetex='cray' not in os.environ['MANPATH'])
    plt.rc('font', family='serif')
    plt.rc('font', size=30)
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'legend.fontsize': 20, 'legend.handlelength': 4})
    rcParams.update({'legend.frameon': False})

# OB 170918 ~ added an rads arg which, if true, divides the x axis by pi and relabels axis with pi, pi/2 etc.
def plot_1d(x,y,xlab,title='',ylab='',rads=False): 
    fig = plt.figure(figsize=(12,8))
    if rads:
        xrads = [val*radians for val in x]
        plt.plot(xrads,y,xunits=radians)
    else:
        plt.plot(x,y)
    plt.xlabel(xlab)
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    return fig

# OB ~ Edited to take into account non-uniform grids.
def plot_2d(z,xin,yin,zmin,zmax,xlab='',ylab='',title='',cmp='RdBu',use_logcolor=False, interpolation='nearest'):
    from scipy.interpolate import griddata
    fig = plt.figure(figsize=(12,8))
    nxin,nyin = len(xin),len(yin)
    last_xin, last_yin = xin[nxin-1], yin[nyin-1]
    # Z is likely provided in a 2-d array format. We need to convert this to a single row of length len(xin) * len(yin). TODO CHECK IF Z ALWAYS IN 2D ARRAY FORMAT.
    data_xy = np.zeros((nxin*nyin,2))
    data_values = np.zeros(nxin*nyin)
    for ix in range(nxin):
        for iy in range(nyin):
            data_xy[nxin*iy + ix, 0] = xin[ix]
            data_xy[nxin*iy + ix, 1] = yin[iy]
            data_values[nxin*iy + ix] = z[ix,iy]

    # Generate fine grid on which to interpolate.
    x=np.linspace(xin[0],last_xin,1000)
    y=np.linspace(yin[0],last_yin,1000)
    grid_x, grid_y = np.meshgrid(x,y)           
   
    if use_logcolor:
        color_norm = mcolors.LogNorm(zmin,zmax)
    else:
        color_norm = mcolors.Normalize()
 
    z_interp = griddata(data_xy,data_values,(grid_x,grid_y),method=interpolation)
    plt.imshow(z_interp, cmp, vmin=np.amin(data_values), vmax=np.amax(data_values),
                extent=[x.min(),x.max(),y.min(),y.max()],
                interpolation='nearest', origin='lower', aspect='auto', norm=color_norm)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    return fig

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
