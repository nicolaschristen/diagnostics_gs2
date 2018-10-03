from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import pickle
import copy as cp

import gs2_plotting as gplot

def my_single_task(ifile,run,myin,myout,mygrids,mytime,myfields,stitching=False):

    # Compute and save to dat file
    if not run.only_plot:
        
        # Quantities needed for kperp2
        gds2  = get_dict_item('gds2' , myout) 
        gds21 = get_dict_item('gds21', myout) 
        gds22 = get_dict_item('gds22', myout) 
        
        theta  = get_dict_item('theta' , myout)
        theta0 = get_dict_item('theta0', myout)
        
        bmag = get_dict_item('bmag' , myout)

        shat = myin['theta_grid_parameters']['shat']
        
        ky = mygrids.ky
        kx = mygrids.kx
       
        # Get kperp2
        kperp2 = np.zeros((len(theta), mygrids.nx, mygrids.ny),dtype=float)
        for iky in range(mygrids.ny):
            if ky[iky] == 0.0:
                for ikx in range(mygrids.nx):
                    kperp2[:,ikx,iky] = kx[ikx]**2 * gds22/shat**2
            else:
                for ikx in range(mygrids.nx):
                    kperp2[:,ikx,iky] = ky[iky]**2 * (gds2 + 2.0*theta0[iky,ikx]*gds21 + theta0[iky,ikx]**2 * gds22)
    
        # Drifts
        gbdrift  = get_dict_item('gbdrift',  myout)
        cvdrift  = get_dict_item('cvdrift',  myout)
        cvdrift0 = get_dict_item('cvdrift0', myout)
        gbdrift0 = get_dict_item('gbdrift0', myout)


        # Save computed quantities      OB 140918 ~ added tri,kap to saved dat.
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.along_tube.dat'
        mydict = {'theta':mygrids.theta,'nxmid':mygrids.nxmid,
                'naky':mygrids.ny,'nakx':mygrids.nx,'kx':mygrids.kx,'ky':mygrids.ky,'kperp2':kperp2,
                'time':mytime.time,'time_steady':mytime.time_steady,'it_min':mytime.it_min,'it_max':mytime.it_max,
                'gbdrift':gbdrift, 'cvdrift':cvdrift, 'cvdrift0':cvdrift0, 'gbdrift0':gbdrift0,
                'gds2':gds2, 'gds21':gds21, 'gds22':gds22, 'bmag':bmag
        }
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mydict,datfile)

        # Save time obj
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.time.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mytime,datfile)
        
        # Save grid obj
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.grids.dat'
        with open(datfile_name,'wb') as datfile:
            pickle.dump(mygrids,datfile)

        # OB SAVES DATA TO AN ASCII FILE - UNCOMMENT IF YOU WANT THIS.
        #cols = ["Theta", "bmag", "gds2", "gds21", "gds22", "kperp2", "gbdrift", "gbdrift0", "cvdrift", "cvdrift0"]
        #geo_data = []
        #kperp2_theta = (kperp2.mean(axis=2)).mean(axis=1)
        #for i in range(len(theta)):
        #    geo_data.append([theta[i], bmag[i], gds2[i], gds21[i], gds22[i], kperp2_theta[i], gbdrift[i], gbdrift0[i], cvdrift[i], cvdrift0[i]])
        #import csv
        #with open(run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + "_geom.csv", 'w') as csvfile:
        #    writer = csv.writer(csvfile)
        #    writer.writerow(cols)
        #    writer.writerows(geo_data)

    else:

        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.along_tube.dat'
        with open(datfile_name,'rb') as datfile:
            mydict = pickle.load(datfile)
                            
        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.time.dat'
        with open(datfile_name,'rb') as datfile:
            mytime = pickle.load(datfile)

        datfile_name = run.work_dir + run.dirs[ifile] + run.out_dir + run.files[ifile] + '.grids.dat'
        with open(datfile_name,'rb') as datfile:
            mygrids = pickle.load(datfile)

    if not run.no_plot and not stitching: # plot fluxes for this single file
        plot_along_tube(ifile,run,mytime,mydict)
    
            
def get_dict_item(name, myout, mygrids=None, mytime=None):
    if myout[name+"_present"]:
            value = myout[name]
    else:
            value = np.arange(1,dtype=float)
    return value
 
def plot_along_tube(ifile,run,mytime,mydict):
    
    kperp2_theta_x_y = mydict['kperp2']
    
    gds2  = mydict['gds2' ]
    gds21 = mydict['gds21']
    gds22 = mydict['gds22']

    gbdrift =  mydict['gbdrift']
    gbdrift0 = mydict['gbdrift0']
    cvdrift =  mydict['cvdrift']
    cvdrift0 = mydict['cvdrift0']

    bmag = mydict['bmag']

    pdflist = []    
    tmp_pdf_id = 0
    write = False
    
    if bmag is not None:
        gplot.plot_1d(mydict['theta'],bmag, '$\\theta$', title = 'bmag', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True
      
    if gds2 is not None:
        gplot.plot_1d(mydict['theta'],gds2, '$\\theta$', title = 'gds2', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True
   
    if gds21 is not None:
        gplot.plot_1d(mydict['theta'],gds21, '$\\theta$', title = 'gds21', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True

    if gds22 is not None:
        gplot.plot_1d(mydict['theta'],gds22, '$\\theta$', title = 'gds22', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True
        
    if kperp2_theta_x_y is not None:
        # Average over kx,ky.
        kperp2_theta_x = kperp2_theta_x_y.mean(axis=2)
        kperp2_theta = kperp2_theta_x.mean(axis=1)
        gplot.plot_1d(mydict['theta'],kperp2_theta,'$\\theta$', title='$k_\perp^2$', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True
     
    if gbdrift is not None:
        gplot.plot_1d(mydict['theta'],gbdrift, '$\\theta$', title = 'gbdrift', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True
  
    if gbdrift0 is not None:
        gplot.plot_1d(mydict['theta'],gbdrift0, '$\\theta$', title = 'gbdrift0', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True

    if cvdrift is not None:
        gplot.plot_1d(mydict['theta'],cvdrift, '$\\theta$', title = 'cvdrift', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True
 
    if cvdrift0 is not None:
        gplot.plot_1d(mydict['theta'],cvdrift0, '$\\theta$', title = 'cvdrift0', rads = True, grid="x")
        tmp_pdfname = 'tmp'+str(tmp_pdf_id)
        gplot.save_plot(tmp_pdfname, run, ifile)
        pdflist.append(tmp_pdfname)
        tmp_pdf_id = tmp_pdf_id+1
        write = True

    if write:
        print(run.scan_name)
        merged_pdfname = 'along_tube' + run.scan_name
        gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

   
