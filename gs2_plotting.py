import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PyPDF2 import PdfFileMerger, PdfFileReader
import os

myred = [183./255, 53./255, 53./255]
myblue = [53./255, 118./255, 183./255]

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
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=30)
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'legend.fontsize': 20, 'legend.handlelength': 4})
    rcParams.update({'legend.frameon': False})

def plot_1d(x,y,xlab,title='',ylab=''):

    fig = plt.figure(figsize=(12,8))
    plt.plot(x,y)
    plt.xlabel(xlab)
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    return fig

def plot_2d(z,xin,yin,zmin,zmax,xlab='',ylab='',title='',cmp='RdBu'):

    fig = plt.figure(figsize=(12,8))
    x,y = np.meshgrid(xin,yin)
    plt.imshow(z, cmap=cmp, vmin=zmin, vmax=zmax,
               extent=[x.min(),x.max(),y.min(),y.max()],
               interpolation='nearest', origin='lower', aspect='auto')
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
