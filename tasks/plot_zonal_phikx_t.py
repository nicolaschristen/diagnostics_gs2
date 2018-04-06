# -*- coding: utf-8 -*-
def plot_zonal_phi_kx_vs_t(mygrids, mytime, phiz):

    from matplotlib import pyplot as plt
    import numpy as np
    
    eps = np.finfo(float).eps

    fig=plt.figure(figsize=(12,8))
    for ikx in range(phiz.shape[1]):
        if np.absolute(mygrids.kx_gs2[ikx]) > eps:
            plt.plot(mytime.time,phiz[:,ikx])

    plt.xlabel('$t (v_{t}/a)$')
    plt.xlim([mytime.time[0],mytime.time[mytime.ntime-1]])
    plt.title('$\Phi_{zf}(k_x)$')

    return fig
