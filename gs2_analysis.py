print("before imports")
import gs2_run_parameters as grunpar
import gs2_data as gdata
import gs2_grids as ggrids
import gs2_time as gtime
import gs2_fields as gfields
import gs2_plotting as gplot
import gs2_tasks as gtasks
import os
import copy
import pickle
# TODO: add 'your_task' to the array.
tasks_choices = ['fluxes', 'zonal', 'tcorr', 'flowtest', 'floquet','lingrowth','fluxes_stitch','potential','range_pot']
print("test!")
# Fom command-line arguments, get info about this analysis run (filenames, tasks to complete ...)
run = grunpar.runobj(tasks_choices)

# Empty object in which the user can store variables computed from each single file,
# for each separate task that is being executed.
# Use: full_space[ifile]['name_of_my_task'].myvar = myval
class task_storage: pass
file_space = {task_name: task_storage() for task_name in run.tasks}
full_space = [copy.deepcopy(file_space) for ifile in range(len(run.fnames))]

gplot.set_plot_defaults()

myin = []
myout = []
mygrids = []
mytime = []
myfields = []
mytxt = []


# Loop over all files specified by user
for ifile in range(len(run.fnames)):
    print('Carring out work for run name ' + str(run.fnames[ifile]))
       
    # OB 031018 ~ If output directory does not already exist, then add it. Removes slash at end when searching in listdir.
    if run.out_dir[:len(run.out_dir)-1] not in os.listdir(run.work_dir + run.dirs[ifile]):
        print('Output directory not found, generating directory at ' + run.work_dir + run.dirs[ifile] + run.out_dir)
        os.mkdir(run.work_dir + run.dirs[ifile] + run.out_dir)

    txtname = run.work_dir + run.dirs[ifile] + run.out_dir + 'info_' + run.files[ifile] + '.txt'
    mytxt = open(txtname, 'w')
    
    if not run.only_plot:
        # Get input parameters for this GS2 simulation from .in file
        myin = gdata.get_input(ifile, run)
        # Get output from corresponding .out.nc file
        myout = gdata.get_output(ifile, run)
        # Extract grids from output
        mygrids = ggrids.gridobj(myout)
        
        # Extract time from output
        mytime = gtime.timeobj(myout, run.twin)
        # Extract fields from output
        fields_file_name = run.files[ifile] + '.fields.dat'
        if ( fields_file_name in os.listdir(run.work_dir+run.dirs[ifile] + run.out_dir) ):
            with open(run.work_dir+run.dirs[ifile] + run.out_dir+fields_file_name,'rb') as datfile:
                myfields = pickle.load(datfile)
            print('Loaded fields from previous analysis! Delete .fields.dat file if you want the fields object to be recalculated!')
        else:
            myfields = gfields.fieldobj(myout, mygrids, mytime)
        #myfields = None
    # Loop over all tasks that have been requested by user
    for itask in range(len(run.tasks)):
        task_name = run.tasks[itask]
        # Complete part of tasks that require a single output file.
        # For tasks requiring a set of files (e.g. scans), the user can store
        # variables from this specific simulation into task_space = full_space[ifile]['name_of_my_task'].
        gtasks.complete_task_single(ifile, task_name, run, myin, myout, mygrids, mytime, myfields, mytxt, full_space[ifile][task_name])
        print("completing single task {}".format(task_name))
                

    if not run.only_plot:
        mytxt.close()

# Complete part of tasks that require a set of file,
# using the variables stored into full_space
if len(run.fnames) > 1:
    for itask in range(len(run.tasks)):
        task_name = run.tasks[itask]
        gtasks.complete_task_scan(task_name, run, full_space)
