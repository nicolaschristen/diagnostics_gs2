import sys
import os

# Add path to directory where task-files are stored
taskdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tasks')
sys.path.insert(0, taskdir)

import zonal
import fluxes
import time_correlation as tcorr
# TODO: save your python script in the task/ folder and import it here.
import flowtest
import floquet
import lingrowth
import potential
import boxballoon
import real
import wzonal
import range_phi

# Complete part of tasks that require a single in/out pair of files.

# TODO: add a case for 'your_task' in the function complete_task_single below and execute the
#       part of your task that requires a single in/out pair of files.
#
# Available variables to work with:
#
# - task_space: can be used to store quantities from this particular in/out pair for later use (e.g. plotting scans).
#               can be defined to be whatever type the user prefers (dictionary, object, ...).
#
# - ifile: index of in/out pair being currently analysed
#
# - task: name of task being currently executed
#
# - run: object with attributes:
#     - fnames[:]: ndarray of all filenames
#     - work_dir: name of directory with GS2 input/output files
#     - out_dir: name of directory where output of analysis will be saved
#
# - myin['gs2_namelist']['param_name']: dict containing all input parameters specified in .in file.
#
# - myout['var']: dict with GS2 output variables.
#     - more variables can be added to the list in gs2_data.py
#     - corresponding 'var_present' key is True if variable var was found in .out.nc file, False otherwise.
#
# - mygrids: object with attributes related to grids in x,y,kx,ky,theta,vpar (see gs2_grids.py).
#
# - mytime: object with (see gs2_time.py)
#     - attributes related to grids in time and freq
#     - time-averaging function
#
# - myfields: object with attributes related to phi,dens,upar,tpar,tperp and E-field (see gs2_fields.py).
#
# - mytxt: .txt file opened in 'w' mode to print stuff.

def complete_task_single(ifile, task, run, myin, myout, mygrids, mytime, myfields, mytxt, task_space):

    if (task in ['fluxes']):#,'tri_kap_nl']):
        
        fluxes.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if (task == 'fluxes_stitch'):
       
        stitching = True
        fluxes.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields,stitching)

    if (task == 'along_tube'):
        along_tube.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if (task == 'zonal'):

        myzonal = zonal.zonalobj(mygrids, mytime, myfields)
        myzonal.plot(ifile, run, myout, mygrids, mytime)

    if (task == 'tcorr'):

        myzonal = zonal.zonalobj(mygrids, mytime, myfields)
        mytcorr = tcorr.tcorrobj(mygrids, mytime, myfields, myzonal)
        mytcorr.plot(ifile, run, mytime, myfields, mytxt)

    if (task == 'flowtest'):

        flowtest.store(myin, myout, task_space)

    if (task == 'floquet'):

        floquet.my_task_single(ifile, run, myin, myout, mytime, task_space)

    if (task in ['lingrowth','tri_kap_lin' , 'kap_lin', 'bprim_lin', 'lin_compare']):

        lingrowth.my_task_single(ifile, run, myin, myout)

    if (task == 'potential'):

        potential.my_task_single(ifile, run, myin, myout, mygrids)
    
    if (task == 'real'):
        real.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if task == 'wzonal':
        wzonal.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if (task == 'range_pot'):
        range_phi.my_task_single(ifile, run, myin, myout)
    if (task == 'boxballoon'):
        boxballoon.my_task_single(ifile,run,myin,myout)
# Complete part of tasks that require the collection of in/out pairs (e.g. plotting a parameter scan).

# TODO: add a case for 'your_task' in the function complete_task_scan below and execute the
#       part of your task that requires the entire collection of in/out pairs (e.g. plotting a parameter scan).
#
# Available variables:
#
# -- full_space[ifile]['task_name']: dict containing all task_spaces stored by complete_task_single.
#
# -- task: name of task being currently executed
#
# -- run: object with attributes:
#     - fnames[:]: ndarray of all filenames
#     - word_dir: name of directory with GS2 input/output files
#     - out_dir: name of directory where output of analysis will be saved

def complete_task_scan(task, run, full_space):

    if (task in ('basic','fluxes','zonal','tcorr')):
        return
    
    # OB ~ Scan in triangularity and elongation.
    if (task == 'tri_kap_nl'):
        fluxes.trikap(run)
    elif (task == 'beta_kap_lin'):
        lingrowth.betakap(run)
    elif (task == 'tri_kap_lin'):
        lingrowth.trikap(run)
    elif (task=='kap_lin'):
        lingrowth.kap(run)
    elif (task=='bprim_lin'):
        lingrowth.bprim(run)
    if (task == 'lin_compare'):
        lingrowth.compare(run)

    if (task == 'flowtest'):

        flowtest.plot(run, full_space)

    if (task == 'fluxes_stitch'):
       
        fluxes.stitching_fluxes(run)
    if (task == 'compare_phit'):
        fluxes.compare_time_traces(run)
    if (task == 'resolution'):
        fluxes.compare_flux_tavg(run)

    if task == 'box_scan':
        boxballoon.kyscan(run)
    
    elif '_scan' in task:
        if task.count('_') == 1:
            range_phi.scan1d(run,task[:task.index('_')])
            return
        elif task.count('_') == 2:
            x = task[:task.index('_')]
            task = task[(task.index('_')+1):]
            y = task[:task.index('_')]
            range_phi.scan2d(run,x,y)
    elif '_fsscan' in task:
        if task.count('_') == 1:
            floquet.scan1d(run,task[:task.index('_')])
            return
        elif task.count('_') == 2:
            x = task[:task.index('_')]
            task = task[(task.index('_')+1):]
            y = task[:task.index('_')] 
            floquet.scan2d(run,x,y)
    elif 'linflxcompare' in task:
        range_phi.compare_fluxes(run)
    elif 'flxcompare' in task:
        fluxes.compare_fluxes(run)
    elif '_compare' in task:
        task = task[:task.rfind('_')]
        jobs = []
        sameplot = []        
        while '_' in task or '-' in task:
            # Sort frome end to avoid -1 errors from task.find()
            ichar = max(task.rfind('-'), task.rfind('_'))
            split = task[ichar] == '_'
            sameplot.append(task[(ichar+1):])
            task = task[:ichar]
            if split:
                jobs.append(sameplot[::-1])
                sameplot = []
        sameplot.append(task)
        jobs.append(sameplot[::-1])
        range_phi.compare(run, jobs[::-1])
    elif '_nlcompare' in task:
        task = task[:task.rfind('_')]
        jobs = []
        sameplot = []        
        while '_' in task or '-' in task:
            # Sort frome end to avoid -1 errors from task.find()
            ichar = max(task.rfind('-'), task.rfind('_'))
            split = task[ichar] == '_'
            sameplot.append(task[(ichar+1):])
            task = task[:ichar]
            if split:
                jobs.append(sameplot[::-1])
                sameplot = []
        sameplot.append(task)
        jobs.append(sameplot[::-1])
        fluxes.compare(run, jobs[::-1])

