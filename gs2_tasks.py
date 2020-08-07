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
import linbox
import linrange
import potential
import boxballoon
import fields_real_space


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

    if (task == 'fluxes'):
       
        fluxes.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if (task == 'fluxes_stitch'):
       
        stitching = True
        fluxes.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields,stitching)

    if (task == 'zonal'):

        myzonal = zonal.zonalobj(mygrids, mytime, myfields)
        myzonal.plot(ifile, run, myout, mygrids, mytime)

    if (task == 'tcorr'):

        myzonal = zonal.zonalobj(mygrids, mytime, myfields)
        mytcorr = tcorr.tcorrobj(mygrids, mytime, myfields, myzonal)
        mytcorr.plot(ifile, run, mytime, myfields, mytxt)

    if (task == 'flowtest'):

        flowtest.store(myin, myout, task_space)

    if (task == 'floquet' or task == 'linbox'):

        linbox.my_task_single(ifile, run, myin, myout, mytime, task_space)

    if (task == 'linrange'):

        linrange.my_task_single(ifile, run, myin, myout, mytime)

    if (task == 'potential'):

        potential.my_task_single(ifile, run, myin, myout, mygrids, mytime)

    if (task == 'boxballoon'):

        boxballoon.my_task_single(ifile, run, myin, myout)

    if (task == 'fields_real_space'):

        fields_real_space.my_task_single(ifile, run, myin, myout, mygrids)

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

    if (task == 'flowtest'):

        flowtest.plot(run, full_space)

    if (task == 'floquet_scan' or task == 'linbox_scan'):

        linbox.task_scan(run, full_space)

    if (task == 'fluxes_stitch'):
       
        fluxes.stitching_fluxes(run)
