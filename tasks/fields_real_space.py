import gs2_plotting as gplot
import gs2_fft as gfft



# vvv USER PARAMETERS vvv

# Plot in lab frame (True) or shearing frame (False)
in_lab_frame = False

# ^^^ USER PARAMETERS ^^^



def my_single_task(ifile, run, myin, myout, mygrids):

    # Electrostatic potential, at ig=igomega (outboard midplane by default)
    phi = myout['phi_igomega_by_mode']

    # Inverse Fourier transform to direct space
    ky_full, phi_full, phi_full_fft = gfft.gs2_fft( phi, mygrids.nx, mygrids.ny, mygrids.ky,
                                                    kx_shift = myout['kx_shift'],
                                                    g_exb = myin['dist_fn_knobs']['g_exb'],
                                                    t = myout['t'][-1],
                                                    lab_frame = in_lab_frame )

    # Plotting

    tmp_pdf_id = 1
    pdflist = []
    tmp_pdf_id_fromSum = 1
    pdflist_fromSum = []

    glot.plot_2d( phi_full_fft, mygrids.xgrid, mygrids.ygrid,
                  np.amin(phi_full_fft), np.amax(phi_full_fft),
                  xlab = '$x$ [$\\rho_i$]', ylab = '$y$ [$\\rho_i$]',
                  title = '$\\varphi$',
                  cmp = 'RdBu_c' )

    tmp_pdfname = 'tmp'+str(tmp_pdf_id)
    plt.savefig('postproc/'+tmp_pdfname+'.pdf')
    pdflist.append(tmp_pdfname)
    tmp_pdf_id = tmp_pdf_id+1

    # Merge pdfs and save
    merged_pdfname = 'fields_real_space'
    gplot.merge_pdfs(pdflist, merged_pdfname, run, ifile)

