# Button: to check

""" This module performs basic resting-state fMRI """

import os, os.path as op
import sys
from time import time
from ...logme import *
import nibabel as nib

def fmri2nifti_unpack():
    """ convert DICOM to NIFTI
    """
    nifti_dir = op.join(gconf.get_nifti())
    fmri_dir = gconf.get_rawrsfmri()

    log.info("Convert rsfMRI ...")
    # check if .nii.gz / .nii.gz is already available
    if op.exists(op.join(fmri_dir, 'fMRI.nii.gz')):
        shutil.copy(op.join(dsi_dir, 'fMRI.nii.gz'), op.join(nifti_dir, 'fMRI.nii.gz'))
    else:
        # read data
        first = gconf.get_dicomfiles('fMRI')[0]
        diff_cmd = 'diff_unpack %s %s' % (first,
                                 op.join(nifti_dir, 'fMRI.nii.gz'))
        runCmd(diff_cmd, log)

def ralign():
    """ realign volume with mc flirt
    """
    param = ''
    flirt_cmd = 'mcflirt -in %s %s' % (
            op.join(gconf.get_nifti(), 'fMRI.nii.gz'),
            param)
    runCmd(flirt_cmd, log)

def mean_fmri():
    """ compute mean fMRI
    """
    nifti_dir = op.join(gconf.get_nifti())

    # compute mean
    a = nib.load( op.join(nifti_dir, 'fMRI.nii.gz') )
    ad = a.get_data()
    hdr = a.get_hdr()

    amean = ad.mean( axis = 3 )

    nib.save( op.join(nifti_dir, 'meanfMRI.nii.gz'), nib.Nifti1Image( amean, a.get_affine(), hdr ) )

    
def register_t1_to_meanfmri():
    """ register T1 to mean fMRI
    """
    param = '-dof 6 -cost mutualinfo'
    flirt_cmd = 'flirt -in %s -ref %s -out %s -omat %s %s' % (
            op.join(gconf.get_nifti(), 'T1.nii.gz'),
            op.join(gconf.get_nifti(), 'meanfMRI.nii.gz'),
            op.join(gconf.get_nifti(), 'T1-TO-fMRI.nii.gz'),
            op.join(gconf.get_nifti_trafo(), 'T1-TO-fMRI.mat'),
            param)
    runCmd(flirt_cmd, log)

    
def apply_registration_roi_to_fmean():
    """ apply registration ROI_HR to fmean
    """
    outmat = op.join( gconf.get_nifti_trafo(), 'T1-TO-fMRI.mat' )

    param = '-interp nearestneighbour'
    for s in gconf.parcellation.keys():
        outfile = op.join(gconf.get_cmp_fmri(), 'ROI_HR_th-TO-fMRI-%s.nii.gz' % s)
        flirt_cmd = 'flirt -applyxfm -init %s -in %s -ref %s -out %s %s' % (
                    outmat,
                    roifile,
                    op.join(gconf.get_nifti(), 'meanfMRI.nii.gz'),
                    outfile,
                    param)

        runCmd( flirt_cmd, log )

        if not op.exists(outfile):
            msg = "An error occurred. File %s not generated." % outfile
            log.error(msg)
            raise Exception(msg)

    log.info("[ DONE ]")
    

def average_rsfmri():
    """ t[a==1].mean(axis=0): matrix output: nrROI x T save matrix in CMP/fMRI
    """
    fdata = nib.load( op.join(gconf.get_nifti(), 'fMRI.nii.gz') ).get_data()

    tp = fdata.shape[3]

    for s in gconf.parcellation.keys():
        infile = op.join(gconf.get_cmp_fmri(), 'ROI_HR_th-TO-fMRI-%s.nii.gz' % s)
        mask = nib.load( infile ).get_data().astype( np.uint32 )

        N = mask.max()
        # matrix number of rois vs timepoints
        odata = np.zeros( (N,tp), dtype = np.float32 )

        for i in range(1,N+1):
            fdata[i,:] = fdata[mask==i].mean( axis = 0 )

        np.save( op.join(gconf.get_cmp_fmri(), 'averageTimeseries_%s.npy' % s) )


def run(conf):
    """ Run the first rsfmri analysis stage

    Parameters
    ----------
    conf : PipelineConfiguration object

    """
    # setting the global configuration variable
    globals()['gconf'] = conf
    globals()['log'] = gconf.get_logger()
    start = time()

    log.info("resting state fMRI stage")
    log.info("========================")

    fmri2nifti_unpack()
    ralign()
    mean_fmri()
    register_t1_to_meanfmri()
    apply_registration_roi_to_fmean()
    average_rsfmri()

    log.info("Module took %s seconds to process." % (time()-start))

    if not len(gconf.emailnotify) == 0:
        msg = ["rsfMRI analysis", int(time()-start)]
        send_email_notification(msg, gconf, log)

def declare_inputs(conf):
    """Declare the inputs to the stage to the PipelineStatus object"""

    # require dicom files
    first = conf.get_dicomfiles( 'fMRI')[0]
    pat, file = op.split(first)
    conf.pipeline_status.AddStageInput(stage, pat, file, 'fmri-dcm')

    # requirements: NativeFreesurfer, and output of parcellation stage...


def declare_outputs(conf):
    """Declare the outputs to the stage to the PipelineStatus object"""

    stage = conf.pipeline_status.GetStage(__name__)

    conf.pipeline_status.AddStageOutput(stage, conf.get_nifti(), 'meanfMRI.nii.gz', 'meanfmri-nii-gz')


