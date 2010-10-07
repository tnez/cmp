import os, os.path as op
from time import time
#import logging
#log = logging.getLogger()
from glob import glob
import subprocess

import re
import sys
import numpy as np
import pickle

import struct
import math
import nibabel

import pickle
import math

################################################################################
def mm2index(mm3, hdrStreamline):
    """ 
    Function
    ----------
    Get the matrix index from mm
        
    Inputs
    ----------
    mm3: [X, Y, Z] in mm
    hdrStreamline: header of the fibers.trk
    
    Outputs
    ----------
    index: [X, Y, Z] 
    
    Date
    ----------
    2010-09-10
    
    Author
    ----------
    Christophe Chenes
    """
    
    index = np.zeros(3)
    index[0] = int(math.ceil( mm3[0] / hdrStreamline['voxel_size'][0] - hdrStreamline['voxel_size'][0]/2 )) # round => math.ceil
    index[1] = int(math.ceil( mm3[1] / hdrStreamline['voxel_size'][1] - hdrStreamline['voxel_size'][1]/2 )) # 0.5 => hdrStreamline['voxel_size'][i]/2
    index[2] = int(math.ceil( mm3[2] / hdrStreamline['voxel_size'][2] - hdrStreamline['voxel_size'][2]/2 ))
    index[index<0] = 0
    if index[0]>hdrStreamline['dim'][0]:
        index[0] = hdrStreamline['dim'][0]
    if index[1]>hdrStreamline['dim'][1]:
        index[1] = hdrStreamline['dim'][1]
    if index[2]>hdrStreamline['dim'][2]:
        index[2] = hdrStreamline['dim'][2]
    return index
################################################################################


################################################################################
def getValFromScalarMap(mm3, scalar, hdr):
    """ 
    Function
    ----------
    Get the value from the scalar map according to the index in mm
        
    Inputs
    ----------
    mm3: [X, Y, Z] in mm
    scalar: nifti image
    hdrStreamline: header of the fibers.trk
    
    Outputs
    ----------
    val: scalar value 
    
    Date
    ----------
    2010-09-13
    
    Author
    ----------
    Christophe Chenes
    """
    
    # TODO check the hdr from scalar and fibers
    index = mm2index(mm3, hdr)
    val = scalar.get_data()[index[0], index[1], index[2]]
    return val     
################################################################################


################################################################################
def DTB__load_endpoints_from_trk(fib, hdr):
    """ 
    Function
    ----------
    Get the endpoints from each fibers
        
    Inputs
    ----------
    fib: the fibers data
    hdr: the header of the fibers.trk
    
    Outputs
    ----------
    endpoints: matrix of size [#fibers, 2, 3] containing for each fiber the
               index of its first and last point
    epLen: array of size [#fibers] containing the length of each fiber
    
    Date
    ----------
    2010-08-20
    
    Authors
    ----------
    Christophe Chenes, Stephan Gerhard
    """

    log.info("============================")
    log.info("DTB__load_endpoints_from_trk")
    
    # Init
    endpoints = np.zeros( (hdr['n_count'], 2, 3) )
    epLen     = []
    pc        = -1

    # Computation for each fiber
    for n in range(0, hdr['n_count']):
    
        # Percent counter
        pcN = int(round( float(100*n)/hdr['n_count'] ))
        if pcN > pc and pcN%10 == 0:	
            pc = pcN
            print '\t\t%4.0f%%' % (pc)

        # Get the first and last point for the current fiber
        dataFiber = np.array(fib[n][0])
        dataFirst = dataFiber[0]
        dataLast  = dataFiber[dataFiber.shape[0]-1]		      
		
        # Translate from mm to index
        endpoints[n,0,:] = mm2index(dataFirst, hdr)	
        endpoints[n,1,:] = mm2index(dataLast, hdr)		    
        epLen.append(dataFiber.shape[0]-1)
		
    # Return the matrices  
    return endpoints, epLen  
	
    log.info("done")
    log.info("============================")
    print '\n###################################################################\n'
################################################################################


################################################################################
def DTB__cmat_shape(fib, hdr):
    """ 
    Function
    ----------
    Get some shape informations from the fibers:
    length
        
    Inputs
    ----------
    fib: the fibers data
    hdr: the header of the fibers.trk
    
    Outputs
    ----------
    cmat_shape.npy file in matrices containing a matrix of size [#fibers, #infos]
    
    Date
    ----------
    2010-08-20
    
    Author
    ----------
    Christophe Chenes
    """
    
    log.info("===============")
    log.info("DTB__cmat_shape")
       	    
    # Get the endpoints for each fibers
    log.info("-------------")
    log.info("Get endpoints")
    en_fname  = op.join(gconf.get_cmt_fibers4subject(sid), 'TEMP_endpoints.npy')
    ep_fname  = op.join(gconf.get_cmt_fibers4subject(sid), 'TEMP_epLen.npy')
    if    not os.path.isfile(en_fname) or not os.path.isfile(ep_fname):
        print 'computing endpoints'
        endpoints, epLen = DTB__load_endpoints_from_trk(fib, hdr)
        print 'saving endpoints'
        np.save(en_fname, endpoints)
        np.save(ep_fname, epLen)
    else:
        print 'loading endpoints'
        endpoints = np.load(en_fname)
        epLen     = np.load(ep_fname)
    log.info("-------------")

	# Used shape informations
    # TODO Where should I get this to avoid hardcoded variable ?
    infos    = ['length'] 
    nbInfos  = 1
    stepSize = 0.5
    
    # Get the shape's informations
    log.info("---------------------")
    log.info("Get the shape's infos")
    vxDim = hdr['voxel_size'][0]

    # Output shape
    out_mat_OLD = np.zeros((endpoints.shape[0], nbInfos))
    out_mat = np.zeros((hdr['n_count'], nbInfos))

    # Euclidian distance
    log.info("Compute the euclidian distance")
    pc = -1
    for i in range(0, hdr['n_count']):  
     
        # Percent counter
        pcN = int(round( float(100*i)/hdr['n_count'] ))
        if pcN > pc and pcN%10 == 0:	
            pc = pcN
            print '\t\t%4.0f%%' % (pc)
            
        # Computation for the current fiber
        crtFiber  = np.array(fib[i][0])
        crtLength = 0
        for j in range(0, crtFiber.shape[0]-1):
            crtEuclidian = math.sqrt( (crtFiber[j+1][0] - crtFiber[j][0])**2 + \
                                      (crtFiber[j+1][1] - crtFiber[j][1])**2 + \
                                      (crtFiber[j+1][2] - crtFiber[j][2])**2 )
            crtLength = crtLength + crtEuclidian    
        out_mat[i] = crtEuclidian        
        
    # For each fibers
    for f in range(0, endpoints.shape[0]):

        # Length
        out_mat_OLD[f,0] = (epLen[f]-1)*stepSize*vxDim;
	
    log.info("---------------------")
    
    # Save the matrix
    log.info("---------------------")
    log.info("Save the shape matrix")
    filepath = op.join(gconf.get_cmt_matrices4subject(sid), 'cmat_shape.npy')
    filepath_OLD = op.join(gconf.get_cmt_matrices4subject(sid), 'OLD_cmat_shape.npy')
    np.save(filepath, out_mat)
    np.save(filepath_OLD, out_mat_OLD)
    log.info("---------------------")
    
    log.info("done")
    log.info("===============")
################################################################################


################################################################################
def DTB__cmat_scalar(fib, hdr):
    """ 
    Function
    ----------
    Get some scalar informations from the fibers and nifti files
        
    Inputs
    ----------
    fib: the fibers data
    hdr: the header of the fibers.trk
    
    Outputs
    ----------
    cmat_scalar1.npy: for each nifti file provided, save a matrix of size [#fibers, 4]
                      containing the mean, min, max and std for each fiber
    ...
    cmat_scalarN.npy: same
    
    Date
    ----------
    2010-09-05
    
    Author
    ----------
    Christophe Chenes
    """
    
    log.info("================")
    log.info("DTB__cmat_scalar")

    # Number of informations: mean max min std
    nInfo = 4
    inPath = gconf.get_cmt_fibers4subject(sid)

    # For each file in the scalar dir
    # TODO Change the method to get scalarfiles
    print '#-----------------------------------------------------------------#\r'
    print '# Scalar informations:                                            #\r'
    scalarDir   = gconf.get_cmt_scalars4subject(sid)
    scalarFiles = np.array(os.listdir(scalarDir))
    nbScalar    = scalarFiles.size
    print nbScalar
    for i in range(0, nbScalar):
        if (scalarFiles[i] != '.' and scalarFiles[i] != '..' and scalarFiles[i] != '.svn') :
            crtName = re.search('[a-z,0-9,A-Z]*.nii',scalarFiles[i]).group(0)
            crtName = re.sub('.nii','',crtName)
            print '\t#'+str(i+1)+' = '+crtName
			
            # Open the file
            scalar = nibabel.load(scalarDir+scalarFiles[i])
			
		    # Create the matrix
            fMatrix = np.zeros((hdr['n_count'],nInfo))
			
            # For each fiber
            # TODO This function is really slow !!
            for j in range(0, hdr['n_count']):
			
				# Get the XYZ in mm
                data = np.array(fib[j][0])
                
                # Get the scalar value from XYZ in mm
                for k in range (0, data.shape[0]):
                    val = getValFromScalarMap(data[k], scalar, hdr)
				
				# Store these informations		
                fMatrix[j, 0] = val.mean() 	
                fMatrix[j, 1] = val.min()				
                fMatrix[j, 2] = val.max()			
                fMatrix[j, 3] = val.std()
				
			# Save the matrix in a file
            print '\tSave'
            filename = 'cmat_'+crtName+'.npy'
            filepath = inPath #gconf.get_cmt_fibers4subject(sid)
            np.save(op.join(filepath, 'matrices/', filename), fMatrix)
            
    log.info("done")
    log.info("================")
################################################################################


################################################################################
def DTB__cmat(fib, hdr): 
    """ 
    Function
    ----------
    Get some scalar informations from the fibers and nifti files
        
    Inputs
    ----------
    fib: the fibers data
    hdr: the header of the fibers.trk
    
    Outputs
    ----------
    cmat_res1.npy: for each ROI nifti file provided, save a connection matrix 
    ...
    cmat_resN.npy: same
    
    Date
    ----------
    2010-09-10
    
    Author
    ----------
    Christophe Chenes
    """
    
    log.info("=========")
    log.info("DTB__cmat")	
          	
    # Get the endpoints for each fibers
    log.info("-------------")
    log.info("Get endpoints")
    en_fname  = op.join(gconf.get_cmt_fibers4subject(sid), 'TEMP_endpoints.npy')
    ep_fname  = op.join(gconf.get_cmt_fibers4subject(sid), 'TEMP_epLen.npy')
    if    not os.path.isfile(en_fname) or not os.path.isfile(ep_fname):
        log.info('computing endpoints')
        endpoints, epLen = DTB__load_endpoints_from_trk(fib, hdr)
        log.info('saving endpoints')
        np.save(en_fname, endpoints)
        np.save(ep_fname, epLen)
    else:
        log.info('loading endpoints')
        endpoints = np.load(en_fname)
        epLen     = np.load(ep_fname)
    log.info("-------------")
	
    # For each resolution
    # TODO We have to give each ROI filepath and it'll compute for it
    log.info("--------------------")
    log.info("Resolution treatment")
    resolution = gconf.parcellation.keys()
    cmat = {} # NEW matrix
    for r in resolution:
        log.info("\tresolution = "+r)
      
        # Open the corresponding ROI
        log.info("\tOpen the corresponding ROI")
        roi_fname = op.join(gconf.get_cmt_fsout4subject(sid), 'registered', 'HR__registered-T0-b0', r, 'ROI_HR_th.nii')
        roi       = nibabel.load(roi_fname)
        roiData   = roi.get_data()
      
        # Create the matrix
        log.info("\tCreate and init the connection matrix")
        n        = roiData.max()
        matrix   = np.zeros((n,n)) # TEST
        arrNodes = np.ndarray((n), object)   
        matEdges = np.ndarray((n,n), object)
        eID      = 0 
        for i in range(0, n): 
            arrNodes[i] = {'node_id': i} 
            for j in range (0,n): 
                matEdges[i][j] = {'edge_id': eID, 'nb_fibers': 0} 
                eID = eID+1
                
        # Open the shape matrix
        log.info("\tLoad the shape matrix")
        shape = np.load(op.join(gconf.get_cmt_matrices4subject(sid), 'cmat_shape.npy'))
        
        # Open each scalar matrix
        # TODO find a better way to work with scalar matrix
        # TODO maybe one big matrix with each scalar info
        # TODO and add in configuration the filename to use
        log.info("\tLoad the scalar matrix => TODO")
#        scalarDir   = gconf.get_cmt_scalars4subject(sid)
#        scalarFiles = np.array(os.listdir(scalarDir))
#        nbScalar    = scalarFiles.size
#        nS          = 0
#        sF          = np.ndarray((nbScalar), 'object')
#        for i in range(0, nbScalar):
#            if (scalarFiles[i] != '.' and scalarFiles[i] != '..' and scalarFiles[i] != '.svn') :
#                sF[nS] = scalarFiles[i]
#                nS = nS+1
#        scalar = np.ndarray((nS), 'object')
#        for i in range(0, nS):
#            scalar[i] = np.load(sF[i])
            
        # For each fiber
        for i in range(0, hdr['n_count']):
         
            # TEMP Add in the corresponding cell the number of fibersFile
            roiF = roiData[endpoints[i, 0, 0], endpoints[i, 0, 1], endpoints[i, 0, 2]]
            roiL = roiData[endpoints[i, 1, 0], endpoints[i, 1, 1], endpoints[i, 1, 2]]
            
            # Filter
            if roiF != 0 and roiL != 0 and roiF != roiL: 
            
                matrix[roiF-1, roiL-1]                += 1 # TEST
                matEdges[roiF-1][roiL-1]['nb_fibers'] += 1 
                
                # Add informations about the current fiber to the corresponding edge
                if matEdges[roiF-1][roiL-1]['nb_fibers'] == 1:
                    matEdges[roiF-1][roiL-1].update({'length': shape[i]})
                    matEdges[roiF-1][roiL-1].update({'fibersId': i})
                else:
                    matEdges[roiF-1][roiL-1]['length']   = np.hstack(( matEdges[roiF-1][roiL-1]['length'], shape[i] ))
                    matEdges[roiF-1][roiL-1]['fibersId'] = np.hstack(( matEdges[roiF-1][roiL-1]['fibersId'], i ))
                
        
        # Add to cmat
        cmat.update({r: {'filename': roi_fname, 'node': arrNodes, 'edge': matEdges}})
        
        # TEST
        filename = 'cmat_'+r+'.npy'
        filepath = op.join(gconf.get_cmt_matrices4subject(sid), filename)
        np.save(filepath, matrix)
        
    log.info("--------------------")

    # NEW matrix => save
    f = open(op.join(gconf.get_cmt_matrices4subject(sid), 'CMat.dat'),'w')
    pickle.dump(cmat,f)
    f.close()

    log.info("done")
    log.info("=========")							
################################################################################


################################################################################
def run(conf, subject_tuple):
    """ 
    Run the connection matrix step
        
    Parameters
    ----------
    conf : PipelineConfiguration object
    subject_tuple : tuple, (subject_id, timepoint)
      Process the given subject
       
    """
    
    # setting the global configuration variable
    globals()['gconf'] = conf
    globals()['sid']   = subject_tuple
    start              = time()
    globals()['log']   = conf.get_logger4subject(subject_tuple)
    
    log.info("########################")
    log.info("Connection matrix module")
    
    # Read the fibers one and for all
    log.info("===============")
    log.info("Read the fibers")
    fibFilename = op.join(gconf.get_cmt_fibers4subject(sid), 'streamline.trk')
    fib, hdr    = nibabel.trackvis.read(fibFilename, False)
    log.info("done")
    log.info("===============")
    
    # Call
    DTB__cmat_shape(fib, hdr)
#    DTB__cmat_scalar(fib, hdr)
#    DTB__cmat(fib, hdr)
    
    log.info("Connection matrix module took %s seconds to process" % (time()-start))
    log.info("########################")
################################################################################
################################################################################
