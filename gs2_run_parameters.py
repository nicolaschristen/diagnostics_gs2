import sys
import argparse
import gs2_quick_parameters as qpar
import os


# TODO: add 'your_task' to the array for single or multi file analysis.
single_tasks_choices = ['fluxes', 'zonal', 'tcorr', 'flowtest', 'floquet','lingrowth','fluxes_stitch','potential']
multi_tasks_choices = ['stitch']


# Split path to single in/out files into dir and name
def splitRunPath(runfilepath):

    lastSlash = runfilepath.rfind('/')
    if lastSlash > 0:
        runDir = runfilepath[:lastSlash+1]
        runName = runfilepath[lastSlash+1:]
    else:
        runDir = ''
        runName = runfilepath
    return runDir,runName


class runobj:

    def __init__(self):

        # Get parameters from command line
        args = self.get_commandline(single_tasks_choices, multi_tasks_choices)

        # If we are in quick mode, get parameters from gs2_quick_parameters.py
        if args.quick:
            
            try:

                self.work_dir = qpar.work_dir

                self.single_files = qpar.single_files
                
                self.single_tasks = qpar.single_tasks
                
                self.single_outdirname = qpar.single_outdirname

                self.multi_tasks = qpar.multi_tasks
                
                self.multi_fname = qpar.multi_fname
                
                self.no_plot = qpar.no_plot
                
                self.only_plot = qpar.only_plot
                
                self.twin = qpar.twin

            except:
                sys.exit('ERROR: please specify all input parameters in gs2_quick_parameters')
        
        # Otherwise use those from command line
        else:

            self.work_dir = args.work_dir
            if self.work_dir is None: self.work_dir = './'

            self.single_files = args.single_files
            if self.single_files is None:
                sys.exit('ERROR: please specify what files to analyse (for help, use option -h).')

            self.single_tasks = args.single_tasks
            if self.single_tasks is None: self.single_tasks = ['fluxes']

            self.single_outdirname = args.single_outdirname
            if self.single_outdirname is None: self.single_outdirname = ''

            self.multi_tasks = args.multi_tasks
            if self.multi_tasks is None: self.multi_tasks = []

            self.multi_outfname = args.multi_outfname
            if self.multi_fname is None: self.multi_fname = 'multi_postproc'

            self.no_plot = args.no_plot

            self.only_plot = args.only_plot

            self.twin = args.twin
            if self.twin is None: self.twin = 0.5

        # Adjust parameters to right format
        if not (self.work_dir[-1]=='/'): self.work_dir = self.work_dir + '/'
        
        if type(self.single_files)==str: self.single_files=[self.single_files]
        
        if type(self.single_tasks)==str: self.single_tasks=[self.single_tasks]
        for itask in range(len(self.single_tasks)):
            if not any(self.single_tasks[itask]==defined_task for defined_task in single_tasks_choices):
                sys.exit('ERROR: single task \'{0}'.format(self.single_tasks[itask]) \
                        + '\' is not defined. Please add it to the list of possible tasks in gs2_run_parameters.py, or specify valid tasknames.')
        
        if len(self.single_outdirname)>0 and not (self.single_outdirname[-1]=='/'): self.single_outdirname = self.single_outdirname + '/'
        
        if type(self.multi_tasks)==str: self.multi_tasks=[self.multi_tasks]
        for itask in range(len(self.multi_tasks)):
            if not any(self.multi_tasks[itask]==defined_task for defined_task in multi_tasks_choices):
                sys.exit('ERROR: multi task \'{0}'.format(self.multi_tasks[itask]) \
                        + '\' is not defined. Please add it to the list of possible tasks in gs2_run_parameters.py, or specify valid tasknames.')

        # If single and multi folders do not exist yet, create them
        for ifile in range(len(self.single_files)):
            run_dir, run_name = splitRunPath(self.single_files[ifile])
            if not os.path.isdir(self.work_dir+run_dir): os.makedirs(self.work_dir+run_dir)
        if len(self.multi_tasks)>0:
            if not os.path.isdir(self.work_dir+self.multi_fname): os.makedirs(self.work_dir+self.multi_fname)


    def full_in_fname(self, ifile):

        run_dir, run_name = splitRunPath(self.single_files[ifile])
        
        return self.work_dir + run_dir + run_name + '.in'


    def full_nc_fname(self, ifile):

        run_dir, run_name = splitRunPath(self.single_files[ifile])
        
        return self.work_dir + run_dir + run_name + '.out.nc'


    def full_single_fname(self, ifile, fextension):

        run_dir, run_name = splitRunPath(self.single_files[ifile])
        
        return self.work_dir + run_dir + self.single_outdirname + run_name + '.' + fextension


    def full_multi_fname(self, fextension):
        
        return self.work_dir + self.multi_fname + '.' + fextension


    def get_commandline(self, single_tasks_choices, multi_tasks_choices):

        parser = argparse.ArgumentParser(description = 'Creating various plots from GS2 NETCDF output files.')
        
        parser.add_argument('-w', '--work_dir', nargs = '?', 
                help = 'Path to working directory (all other paths have to be specified relatively to it). Default is current directory.')
        
        parser.add_argument('-f', '--files', nargs = '*',
                help = '\'path/from/work_dir/to/sim_file\' without extension of file(s). Can be a single str or an array of str. Must be specified.')
        
        parser.add_argument('-t', '--single_tasks', nargs = '*', choices = single_tasks_choices,
                help = 'Task(s) to complete for each single file. Can be a single str or an array of str. Default is \'fluxes\'. Check gs2_run_parameters.py for all possibilities.')
        
        parser.add_argument('-o', '--single_outdirname', nargs = '?',
                help = 'Name of directory where plots will be saved. Must be located in same directory as in/out files. ' \
                + 'Will be created if no such dir. Default creates files in directory where in/out files are located.')
        
        parser.add_argument('-T', '--multi_tasks', nargs = '*', choices = multi_tasks_choices,
                help = 'Task(s) to complete using the whole collection of files (e.g. for scans). None selected by default. Check gs2_run_parameters.py for all possibilities.')
        
        parser.add_argument('-F', '--multi_fname', nargs = '?',
                help = '\'path/from/work_dir/to/multi_file\' to use when saving multi-file analysis, without any extension. Default is \'multi_postproc\'.')

        parser.add_argument('-n', '--no_plot', action = 'store_true', default = False,
                help = 'Saving postproc results to .dat files, without producing plots (useful on HPCs where LaTeX is not installed).')

        parser.add_argument('-p', '--only_plot', action = 'store_true', default = False,
                help = 'Output from tasks has already been saved to .dat files, so no reading from NETCDF files required and only need to plot.')

        parser.add_argument('--twin', nargs = '?',
                help = 'Specify fraction of time over which the solution is nonlinearly saturated (0.0 -> 1.0). Default is 0.5.')

        parser.add_argument('-q', '--quick', action = 'store_true', default = False,
                help = 'Quick mode: all parameters are set to values from gs2_quick_parameters.py file.')
        
        args = parser.parse_args()
        
        return args
