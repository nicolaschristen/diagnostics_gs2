from netCDF4 import Dataset
import numpy as np
import f90nml
import copy as cp

out_varnames = [
        't', 'kx', 'ky',
        'theta', 'theta0',
        'phi',
        # modulus squared, avged over theta, kx and ky [t]
        'phi2',
        # modulus squared, avged over theta [t,ky,kx]
        'phi2_by_mode',
        # modulus squared, avged over theta and kx [t,ky]
        'phi2_by_ky',
        # complex potential fluctuation [t,ky,kx,theta,imag]
        'phi_t',
        # at itheta = igomega (=0 by default) [t,ky,kx,imag]
        'phi_igomega_by_mode',
        # Non-adiabatic part of perturb. dens. at itheta = igomega (=0 by default) [t,spec,ky,kx,imag]
        'density_igomega_by_mode',
        # total perturb. dens. for a given theta, usually outboard mid-plane [t,spec,ky,kx,imag]
        'ntot_igomega_by_mode',
        # complex parallel flow fluctuation for a given theta, usually outboard mid-plane [t,spec,ky,kx,imag]
        'upar_igomega_by_mode',
        # at itheta = igomega (=0 by default) [t,spec,ky,kx,imag]
        'tpar_igomega_by_mode',
        # at itheta = igomega (=0 by default) [t,spec,ky,kx,imag]
        'tperp_igomega_by_mode',
        # total density at last time step [spec,ky,kx,theta,imag]
        'ntot',
        # non-adiabatic part of the density at last time step [spec,ky,kx,theta,imag]
        'density',
        # complex frequency [t,ky,kx,imag]
        'omega',
        # complex frequency avged over navg time-steps [t,ky,kx,imag]
        'omega_average',
        # electrostatic particle flux [t,spec]
        'es_part_flux',
        # electrostatic heat flux [t,spec]
        'es_heat_flux',
        # electrostatic momentum flux [t,spec]
        'es_mom_flux',
        # turbulent energy exchange [t,spec]
        'es_energy_exchange',
        # electrostatic particle flux by k [t,spec,ky,kx]
        'es_part_flux_by_mode',
        # electrostatic heat flux by k [t,spec,ky,kx]
        'es_heat_flux_by_mode',
        # electrostatic momentum flux by k [t,spec,ky,kx]
        'es_mom_flux_by_mode',
        # parallel velocity grid
        'vpa',
        # electrostatic particle flux function of vpa and theta [t,spec,vpa,theta]
        'es_part_sym',
        # electrostatic heat flux function of vpa and theta [t,spec,vpa,theta]
        'es_heat_sym',
        # electrostatic momentum flux function of vpa and theta [t,spec,vpa,theta]
        'es_mom_sym',
        'gds2',
        'gds21',
        'gds22',
        'gbdrift',
        'cvdrift',
        'gbdrift0',
        'cvdrift0',
        'bmag',
        'apar',
        # Cases with flow shear: difference between the time-varying kx in the lab-frame,
        # and the fixed nearest neighbour kx in the shearing frame.
        # [ky]
        'kx_shift',
        'kx_shift_over_time',
        'drhodpsi',
        'zonal_transfer',
        'hypervisc_exp'
        ]

def get_output(ifile, run):

    fname = run.work_dir + run.fnames[ifile] + '.out.nc'
    ncfile = Dataset(fname, 'r')

    myout = {}
    
    # Get variables from NETCDF file and add to myout
    myout['nspec'] = ncfile.dimensions['species'].size
    for varname in out_varnames:
        get_single_output(ncfile, varname, myout)

    # Name of phi_igomega_by_mode used to be phi0, so check for that varname too
    if (not myout['phi_igomega_by_mode_present']):
        get_single_output(ncfile, 'phi0', myout, 'phi_igomega_by_mode')
    
    marconi_crashed = False
    indices_to_delete = [myout['t'].size-2, myout['t'].size-1] # for ollie_badshear_old_id_3
    if(marconi_crashed):
        myout['t'] = np.delete(myout['t'],indices_to_delete)
        myout['phi2'] = np.delete(myout['phi2'],indices_to_delete)
        myout['phi2_by_mode'] = np.delete(myout['phi2_by_mode'],indices_to_delete,axis=0)
        myout['phi_igomega_by_mode'] = np.delete(myout['phi_igomega_by_mode'],indices_to_delete,axis=0)
        myout['ntot_igomega_by_mode'] = np.delete(myout['ntot_igomega_by_mode'],indices_to_delete,axis=0)
        myout['upar_igomega_by_mode'] = np.delete(myout['upar_igomega_by_mode'],indices_to_delete,axis=0)
        myout['tpar_igomega_by_mode'] = np.delete(myout['tpar_igomega_by_mode'],indices_to_delete,axis=0)
        myout['phi2_by_ky'] = np.delete(myout['phi2_by_ky'],indices_to_delete,axis=0)
        myout['omega'] = np.delete(myout['omega'],indices_to_delete,axis=0)
        myout['omega_average'] = np.delete(myout['omega_average'],indices_to_delete,axis=0)
        myout['es_part_flux'] = np.delete(myout['es_part_flux'],indices_to_delete,axis=0)
        myout['es_heat_flux'] = np.delete(myout['es_heat_flux'],indices_to_delete,axis=0)
        myout['es_mom_flux'] = np.delete(myout['es_mom_flux'],indices_to_delete,axis=0)
        myout['es_energy_exchange'] = np.delete(myout['es_energy_exchange'],indices_to_delete,axis=0)
        myout['es_part_flux_by_mode'] = np.delete(myout['es_part_flux_by_mode'],indices_to_delete,axis=0)
        myout['es_heat_flux_by_mode'] = np.delete(myout['es_heat_flux_by_mode'],indices_to_delete,axis=0)
        myout['es_mom_flux_by_mode'] = np.delete(myout['es_mom_flux_by_mode'],indices_to_delete,axis=0)
        myout['es_part_sym'] = np.delete(myout['es_part_sym'],indices_to_delete,axis=0)
        myout['es_heat_sym'] = np.delete(myout['es_heat_sym'],indices_to_delete,axis=0)
        myout['es_mom_sym'] = np.delete(myout['es_mom_sym'],indices_to_delete,axis=0)

    # Adapt for different ways in which kx_shift is output by gs2:
    # myout['kx_shift'] will contain kx_shift at every time-step
    # if kx_shift has indeed been output at every time-step by the code.
    # Otherwise it contains kx_shift at the last time-step.
    if myout['kx_shift_over_time_present'] and myout['kx_shift'].ndim == 1:
        myout['kx_shift'] = cp.deepcopy(myout['kx_shift_over_time'])

    return myout


def get_single_output(ncfile, varname, myout, newname=None):
    
    if (not newname):
        newname = varname
    
    try:
        myout[newname] = np.copy(ncfile.variables[varname][:])
        myout[newname+'_present'] = True
    except KeyError:
        print('INFO: ' + varname + ' not found in netcdf file')
        myout[newname] = None
        myout[newname+'_present'] = False
    
    return


def get_input(ifile, run):
    
    fname = run.work_dir + run.fnames[ifile] + '.in'
    myin = f90nml.read(fname)

    return myin

