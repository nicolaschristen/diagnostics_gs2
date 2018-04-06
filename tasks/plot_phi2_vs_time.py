# -*- coding: utf-8 -*-
import numpy as np

def plot_phi2_kxky_vs_t(mytime,phi2):

    from matplotlib import pyplot as plt

    fig=plt.figure(figsize=(12,8))
    # distinguish between zonal/nonzonal modes
    for ikx in range(phi2.shape[2]-1):
        plt.semilogy(mytime.time,phi2[:,0,ikx+1])

    for iky in range(phi2.shape[1]-1):
        for ikx in range(phi2.shape[2]):
            plt.semilogy(mytime.time,phi2[:,iky+1,ikx])

    plt.xlabel('$t (v_{t}/a)$')
    plt.xlim([mytime.time[0],mytime.time[mytime.ntime-1]])
    plt.title('$\Phi^2(k_x,k_y)$')

    return fig

def plot_phi2_ky_vs_t(mytime, phi2):

    from matplotlib import pyplot as plt

    fig=plt.figure(figsize=(12,8))

    plt.semilogy(mytime.time,phi2[:,0],'--')
    for iky in range(phi2.shape[1]-1):
        plt.semilogy(mytime.time,phi2[:,iky+1])
        
    ## NC ## plot only first ky's with legend
    #mylegend=["ky = 0.000"]
    #for iky in range(5):
        #plt.semilogy(time,phi2[:,iky+1])
        #mylegend=np.append(mylegend,"ky = "+"{:.3f}".format(ky[iky+1]))
    #plt.legend(mylegend,fontsize=16,loc="lower right")
    ## endNC ##

    return fig
