import gs2_storage as gstorage
import numpy as np
import gs2_time as gtime
import copy

def stitch(run):

    # First, create stitched time.dat
    Nfile = len(run.fnames)
    alltimes = [dict() for ifile in range(Nfile)]
    t_stitch = []
    for ifile in range(Nfile):
        fname = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.time.dat'
        gstorage.read_from_file(fname, alltimes[ifile])
        # Remove duplicate points between restart files
        if ifile < Nfile-1:
            t_stitch = np.concatenate((t_stitch, alltimes[ifile]['time'][:-1]))
        else:
            t_stitch = np.concatenate((t_stitch, alltimes[ifile]['time'][:]))

    fname = run.multi_fname + '.time.dat'
    mytime_stitch = gtime.init_and_save_mytime(t_stitch, run.twin, fname)

    # Then create stitched dat file for every single-file task
    for task in run.tasks:
        
        Nfile = len(run.fnames)
        allvars = [dict() for ifile in range(Nfile)]
        stitch_dict = {}

        for ifile in range(Nfile):
            
            # Read dat file for this task and this file into allvars[ifile]
            fname = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.' + task + '.dat'
            gstorage.read_from_file(fname, allvars[ifile])
            Ntime = alltimes[ifile]['ntime']

            for key in list(allvars[ifile].keys()):

                # Find if a dimension corresponds to time
                dim_stitch = None
                varsize = allvars[ifile][key].shape
                for dim in range(len(varsize)):
                    if varsize[dim]==Ntime:
                        # If multiple dimensions of size Ntime there are, very confused I am.
                        if dim_stitch is not None:
                            dim_stitch = int(input('Variable {0}'.format(key) + ' has multiple dimensions of size Ntime. Please indicate which dimension corresponds to time: '))
                            break
                        else:
                            dim_stitch = dim

                # Variables with no time dimension
                if dim_stitch is None:
                    # Re-compute time-averaged quantities
                    print('WARNING: To compute time averaged quantities correctly when stitching files together, please ensure ' ...
                            + 'that those averages are saved with variable names ending in \'_tavg\' when analysing each individual file. ' ...
                            + 'Failing to do so will set the stitched time average to the one of the last individual file.')
                    if key[-5:]=='_tavg':
                        if ifile==0: stitch_dict[key]=0.
                        stitch_dict[key] += allvars[ifile][key] * ...
                        (alltimes[ifile]['time'][alltimes[ifile]['it_max']-1]-alltimes[ifile]['time'][alltimes[ifile]['it_min']]) # NDCQUEST: should be it_max instead ?
                        if ifile == Nfile-1:
                            stitch_dict[key] = stitch_dict[key] / ...
                            (mytime_stitch['time'][mytime_stitch['it_max']-1]-mytime_stitch['time'][mytime_stitch['it_min']]) # NDCQUEST: should be it_max instead ?
                    # Variables with no time-dep: copy from first file
                    elif ifile==0:
                        stitch_dict[key] = copy.deepcopy(allvars[ifile][key])
                
                # Stitch variables that need it
                elif dim_stitch is not None:
                    # Remove duplicate points between restart files
                    # Create array with ndim slice objects
                    myslice = [slice(None)]*allvars[ifile][key].ndim
                    # Exclude last element of stitching dimension
                    myslice[dim_stitch] = slice(None,-1,None)
                    # Convert to tuple to use as ndarray index
                    myslice = tuple(myslice)
                    if ifile == 0:
                        stitch_dict[key] = copy.deepcopy(allvars[ifile][key][myslice])
                    elif ifile < Nfile-1:
                        stitch_dict[key] = np.concatenate((stitch_dict[key], allvars[ifile][key][myslice]), dim_stitch)
                    else:
                        stitch_dict[key] = np.concatenate((stitch_dict[key], allvars[ifile][key]), dim_stitch)
            
            # end of key loop
        
        # end of file loop

        stitch_fname = run.multi_fname + '.' + task + '.dat'
        gstorage.save_to_file(stitch_fname, stitch_dict)
    
    # end of task loop
