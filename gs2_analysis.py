import gs2_run_parameters as grunpar
import gs2_data as gdata
import gs2_grids as ggrids
import gs2_time as gtime
import gs2_fields as gfields
import gs2_plotting as gplot
import gs2_tasks as gtasks
import copy

# Set postprocessing parameters based on command-line arguments
run = grunpar.runobj()

if not run.no_plot:
    gplot.set_plot_defaults()

# Loop over all files specified by user
for ifile in range(len(run.single_files)):

    txtname = run.full_single_fname(ifile, 'info.txt')
    mytxt = open(txtname, 'w')
    
    # Generate basic dictionaries: input, output, grids, time, fields
    if not run.only_plot:
        # Get input parameters for this GS2 simulation from .in file
        myin = gdata.get_input(ifile, run)
        # Get output from corresponding .out.nc file
        myout = gdata.get_output(ifile, run)

        # Extract grids from output
        fname = run.full_single_fname(ifile, 'grids.dat')
        mygrids = ggrids.init_and_save_mygrids(myout, fname)
        # Extract time from output
        fname = run.full_single_fname(ifile, 'time.dat')
        mytime = gtime.init_and_save_mytime(myout['t'], run.twin, fname)
        # Extract fields from output
        fname = run.full_single_fname(ifile, 'fields.dat')
        myfields = gfields.init_and_save_myfields(myout, mygrids, mytime, fname)

    # Loop over all tasks that have been requested by user and analyse a single pair of .in/.out.nc files
    for itask in range(len(run.single_tasks)):
        gtasks.complete_task_single(ifile, itask, run, myin, myout, mygrids, mytime, myfields, mytxt)

    if not run.only_plot:
        mytxt.close()

# Complete part of tasks that require a set of file,
# using the variables stored into full_space
#if len(run.fnames) > 1:
#    for itask in range(len(run.tasks)):
#        task_name = run.tasks[itask]
#        gtasks.complete_task_scan(task_name, run, full_space)
