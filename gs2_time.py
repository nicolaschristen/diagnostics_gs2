import numpy as np
from scipy.integrate import simps
from numpy import fft
from math import ceil
import gs2_storage as gstorage


def init_and_save_mytime(t, twin, fname):

    time = np.copy(t)
    ntime = time.size

    # get starting index for steady-state
    it_min = int(ceil((1.0-twin)*ntime))
    it_max = ntime-1
    # get number of time points in steady-state interval
    it_interval = ntime - it_min

    time_steady = time[it_min:it_max]
    ntime_steady = time_steady.size

    # get set of frequencies sampled by data, assuming equal time steps
    frequency = 2*np.pi*np.fft.fftshift(
            np.fft.fftfreq(ntime_steady,time[ntime-1]-time[ntime-2]))

    wgts_time = np.arange(ntime,dtype=float)
    wgts_time[0] = 0.5*(time[1]-time[0])
    for i in range(1,ntime-1):
        wgts_time[i] = 0.5*(time[i+1]-time[i-1])
    wgts_time[ntime-1] = 0.5*(time[ntime-1]-time[ntime-2])

    del i, t
    mytime = locals()
    del mytime['fname']
    gstorage.save_to_file(fname, mytime)
    return mytime


def timeavg(mytime,ft):
    n = ft.ndim
    sl = [slice(None)] * n
    sl[0] = slice(mytime['it_min'], mytime['it_max'])
    favg = simps(ft[tuple(sl)],x=mytime['time_steady'], axis=0) \
            / (mytime['time'][mytime['it_max']-1]-mytime['time'][mytime['it_min']]) # NDCQUEST: should this be it_max instead ?
    return favg
