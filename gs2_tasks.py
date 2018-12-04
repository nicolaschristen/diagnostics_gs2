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
import along_tube

# Complete part of tasks that require a single pair of .in/.out.nc files.

# TODO: add a case for 'your_task' in the function complete_task_single below and execute the
#       part of your task that requires a single pair of .in/.out.nc files.
#
def complete_task_single(ifile, itask, run, myin, myout, mygrids, mytime, myfields, mytxt):

    if (run.tasks[itask] == 'fluxes'):
       
        fluxes.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if (run.tasks[itask] == 'fluxes_stitch'):
       
        stitching = True
        fluxes.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields,stitching)

    if (run.tasks[itask] == 'along_tube'):
        along_tube.my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields)

    if (run.tasks[itask] == 'zonal'):

        myzonal = zonal.zonalobj(mygrids, mytime, myfields)
        myzonal.plot(ifile, run, myout, mygrids, mytime)

    if (run.tasks[itask] == 'tcorr'):

        myzonal = zonal.zonalobj(mygrids, mytime, myfields)
        mytcorr = tcorr.tcorrobj(mygrids, mytime, myfields, myzonal)
        mytcorr.plot(ifile, run, mytime, myfields, mytxt)

    if (run.tasks[itask] == 'flowtest'):

        flowtest.store(myin, myout, task_space)

    if (run.tasks[itask] == 'floquet'):

        floquet.my_task_single(ifile, run, myin, myout, task_space)

    if (run.tasks[itask] == 'lingrowth'):

        lingrowth.my_task_single(ifile, run, myin, myout)

    if (run.tasks[itask] == 'potential'):

        potential.my_task_single(ifile, run, myin, myout, mygrids)

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
    elif (task == 'tri_kap_lin'):
        lingrowth.trikap(run)

    if (task == 'flowtest'):

        flowtest.plot(run, full_space)

    if (task == 'floquet'):

        floquet.task_scan(run, full_space)

    if (task == 'fluxes_stitch'):
       
        fluxes.stitching_fluxes(run)
