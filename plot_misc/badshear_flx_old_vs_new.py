import sys
import os

# Add path to directory where pygs2 files are stored
maindir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.insert(0, maindir)

import pickle
from matplotlib import pyplot as plt
import gs2_plotting as gplot

# set plotting defaults
gplot.set_plot_defaults()

ispec = 0 # main ion

dirname = 'postproc/'

fname_new = dirname+'ollie_badshear_fexp_stitch'
fname_old = dirname+'ollie_badshear_old_stitch3'

# Read in flux quantities
with open(fname_new+'.fluxes.dat','rb') as datfile:
    fluxdict_new = pickle.load(datfile)
with open(fname_old+'.fluxes.dat','rb') as datfile:
    fluxdict_old = pickle.load(datfile)

vflx_new = fluxdict_new['vflx']
vflx_old = fluxdict_old['vflx']
qflx_new = fluxdict_new['qflx']
qflx_old = fluxdict_old['qflx']
vflx_tavg_new = fluxdict_new['vflx_tavg'][ispec]
qflx_tavg_new = fluxdict_new['qflx_tavg'][ispec]
vflx_tavg_old = fluxdict_old['vflx_tavg'][ispec]
qflx_tavg_old = fluxdict_old['qflx_tavg'][ispec]

# Read in time quantities
with open(fname_new+'.time.dat','rb') as datfile:
    mytime_new = pickle.load(datfile)
with open(fname_old+'.time.dat','rb') as datfile:
    mytime_old = pickle.load(datfile)

time_new = mytime_new.time
time_old = mytime_old.time

##########################
## PLOT MOMENTUM FLUX
##########################

fig=plt.figure(figsize=(12,8))

# indicating area of saturation
plt.axvline(x=mytime_new.time_steady[0], color='grey', linestyle='-')
ax = plt.gca()
ax.axvspan(mytime_new.time_steady[0], max(mytime_new.time_steady[-1],mytime_old.time_steady[-1]), alpha=0.1, color='grey')

# plot time-traces for each species
my_curves = []
my_labels = ['discrete-in-time','continuous-in-time']
crv_old, = plt.plot(mytime_old.time,vflx_old[:,ispec],color=gplot.myblue,linewidth=3.0)
my_curves.append(crv_old)
crv_new, = plt.plot(mytime_new.time,vflx_new[:,ispec],color=gplot.myred,linewidth=3.0)
my_curves.append(crv_new)

plt.xlabel('$t [a/v_{th,i}]$',FontSize='35')
plt.ylabel('$\Pi_i/\Pi_{gB}$',FontSize='35')
plt.xlim([mytime_new.time[0],max(mytime_new.time[-1],mytime_old.time[-1])])
plt.ylim([-0.1,2.0])
plt.grid(True)

my_legend = plt.legend(my_curves,my_labels,frameon=True,fancybox=False,framealpha=1.0,loc='lower right',prop={'size': 28})
my_legend.get_frame().set_facecolor('w')
my_legend.get_frame().set_edgecolor('k')
my_legend.get_frame().set_linewidth(1.0)

# Annotate
xpos = mytime_new.time[-1]*0.05
ypos_new = vflx_tavg_new
note_str_new = 'avg = {:.2f}'.format(vflx_tavg_new)
ypos_old = vflx_tavg_old
note_str_old = 'avg = {:.2f}'.format(vflx_tavg_old)

note_coords = 'data'
note_xy_new = (xpos, ypos_new)
note_xy_old = (xpos, ypos_old)

plt.annotate(note_str_old, xy=note_xy_old, xycoords=note_coords, color=gplot.myblue, \
        fontsize=30, backgroundcolor='w', \
        bbox=dict(facecolor='w', edgecolor=gplot.myblue, alpha=1.0))
plt.annotate(note_str_new, xy=note_xy_new, xycoords=note_coords, color=gplot.myred, \
        fontsize=30, backgroundcolor='w', \
        bbox=dict(facecolor='w', edgecolor=gplot.myred, alpha=1.0))

filename = dirname + 'badshear_vflx_old_vs_new_final' + '.pdf'
plt.savefig(filename)

##########################
## PLOT HEAT FLUX
##########################

fig=plt.figure(figsize=(12,8))

# indicating area of saturation
plt.axvline(x=mytime_new.time_steady[0], color='grey', linestyle='-')
ax = plt.gca()
ax.axvspan(mytime_new.time_steady[0], max(mytime_new.time_steady[-1],mytime_old.time_steady[-1]), alpha=0.1, color='grey')

# plot time-traces for each species
my_curves = []
my_labels = ['discrete-in-time','continuous-in-time']
crv_old, = plt.plot(mytime_old.time,qflx_old[:,ispec],color=gplot.myblue,linewidth=3.0)
my_curves.append(crv_old)
crv_new, = plt.plot(mytime_new.time,qflx_new[:,ispec],color=gplot.myred,linewidth=3.0)
my_curves.append(crv_new)

plt.xlabel('$t [a/v_{th,i}]$',FontSize='35')
plt.ylabel('$Q_i/Q_{gB}$',FontSize='35')
plt.xlim([mytime_new.time[0],max(mytime_new.time[-1],mytime_old.time[-1])])
plt.ylim([-0.1,4.0])
plt.grid(True)

my_legend = plt.legend(my_curves,my_labels,frameon=True,fancybox=False,framealpha=1.0,loc='lower right',prop={'size': 28})
my_legend.get_frame().set_facecolor('w')
my_legend.get_frame().set_edgecolor('k')
my_legend.get_frame().set_linewidth(1.0)

# Annotate
xpos = mytime_new.time[-1]*0.1
ypos_new = qflx_tavg_new
note_str_new = 'avg = {:.2f}'.format(qflx_tavg_new)
ypos_old = qflx_tavg_old
note_str_old = 'avg = {:.2f}'.format(qflx_tavg_old)

note_coords = 'data'
note_xy_new = (xpos, ypos_new)
note_xy_old = (xpos, ypos_old)

plt.annotate(note_str_old, xy=note_xy_old, xycoords=note_coords, color=gplot.myblue, \
        fontsize=30, backgroundcolor='w', \
        bbox=dict(facecolor='w', edgecolor=gplot.myblue, alpha=1.0))
plt.annotate(note_str_new, xy=note_xy_new, xycoords=note_coords, color=gplot.myred, \
        fontsize=30, backgroundcolor='w', \
        bbox=dict(facecolor='w', edgecolor=gplot.myred, alpha=1.0))

filename = dirname + 'badshear_qflx_old_vs_new_final' + '.pdf'
plt.savefig(filename)
