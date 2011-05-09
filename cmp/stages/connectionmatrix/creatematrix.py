# Copyright (C) 2009-2011, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

import os, os.path as op
from time import time
from glob import glob
import numpy as np
import nibabel
import networkx as nx
from ...logme import *
from cmp.util import mean_curvature, length

def compute_curvature_array(fib):
    """ Computes the curvature array """
    log.info("Compute curvature ...")
    
    n = len(fib)
    pc        = -1
    meancurv = np.zeros( (n, 1) )
    for i, fi in enumerate(fib):
        # Percent counter
        pcN = int(round( float(100*i)/n ))
        if pcN > pc and pcN%1 == 0:    
            pc = pcN
            log.info('%4.0f%%' % (pc))
        meancurv[i,0] = mean_curvature(fi[0])

    return meancurv
    log.info("DONE")

def create_endpoints_array(fib, voxelSize):
    """ Create the endpoints arrays for each fiber
        
    Parameters
    ----------
    fib: the fibers data
    voxelSize: 3-tuple containing the voxel size of the ROI image
    
    Returns
    -------
    (endpoints: matrix of size [#fibers, 2, 3] containing for each fiber the
               index of its first and last point in the voxelSize volume
    endpointsmm) : endpoints in milimeter coordinates
    
    """

    log.info("========================")
    log.info("create_endpoints_array")
    
    # Init
    n         = len(fib)
    endpoints = np.zeros( (n, 2, 3) )
    endpointsmm = np.zeros( (n, 2, 3) )
    pc        = -1

    # Computation for each fiber
    for i, fi in enumerate(fib):
    
        # Percent counter
        pcN = int(round( float(100*i)/n ))
        if pcN > pc and pcN%1 == 0:    
            pc = pcN
            log.info('%4.0f%%' % (pc))

        f = fi[0]
    
        # store startpoint
        endpoints[i,0,:] = f[0,:]
        # store endpoint
        endpoints[i,1,:] = f[-1,:]
        
        # store startpoint
        endpointsmm[i,0,:] = f[0,:]
        # store endpoint
        endpointsmm[i,1,:] = f[-1,:]
        
        # Translate from mm to index
        endpoints[i,0,0] = int( endpoints[i,0,0] / float(voxelSize[0]))
        endpoints[i,0,1] = int( endpoints[i,0,1] / float(voxelSize[1]))
        endpoints[i,0,2] = int( endpoints[i,0,2] / float(voxelSize[2]))
        endpoints[i,1,0] = int( endpoints[i,1,0] / float(voxelSize[0]))
        endpoints[i,1,1] = int( endpoints[i,1,1] / float(voxelSize[1]))
        endpoints[i,1,2] = int( endpoints[i,1,2] / float(voxelSize[2]))
        
    # Return the matrices  
    return (endpoints, endpointsmm)  
    
    log.info("done")
    log.info("========================")

def extract(Z, shape, position, fill):
    """ Extract voxel neighbourhood
    Parameters
    ----------
    Z: the original data
    shape: tuple containing neighbourhood dimensions
    position: tuple containing central point indexes
    fill: value for the padding of Z
    Returns
    -------
    R: the neighbourhood of the specified point in Z
    """

    R = np.ones(shape, dtype=Z.dtype) * fill
    P = np.array(list(position)).astype(int)
    Rs = np.array(list(R.shape)).astype(int)
    Zs = np.array(list(Z.shape)).astype(int)

    Z_start = (P - Rs // 2)
    Z_start = (np.maximum(Z_start,0)).tolist()
    Z_stop = (P + Rs // 2) + Rs % 2
    Z_stop = (np.minimum(Z_stop,Zs)).tolist()

    z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
    R = Z[z]

    return R

def save_fibers(oldhdr, oldfib, fname, indices):
    """ Stores a new trackvis file fname using only given indices """

    hdrnew = oldhdr.copy()

    outstreams = []
    for i in indices:
        outstreams.append( oldfib[i] )

    n_fib_out = len(outstreams)
    hdrnew['n_count'] = n_fib_out

    log.info("Writing final no orphan fibers: %s" % fname)
    nibabel.trackvis.write(fname, outstreams, hdrnew)


def cmat(): 
    """ Create the connection matrix for each resolution using fibers and ROIs. """
              
    # create the endpoints for each fibers
    en_fname  = op.join(gconf.get_cmp_fibers(), 'endpoints.npy')
    en_fnamemm  = op.join(gconf.get_cmp_fibers(), 'endpointsmm.npy')
    ep_fname  = op.join(gconf.get_cmp_fibers(), 'lengths.npy')
    curv_fname  = op.join(gconf.get_cmp_fibers(), 'meancurvature.npy')
    intrk = op.join(gconf.get_cmp_fibers(), 'streamline_filtered.trk')

    fib, hdr    = nibabel.trackvis.read(intrk, False)
    
    # Previously, load_endpoints_from_trk() used the voxel size stored
    # in the track hdr to transform the endpoints to ROI voxel space.
    # This only works if the ROI voxel size is the same as the DSI/DTI
    # voxel size.  In the case of DTI, it is not.  
    # We do, however, assume that all of the ROI images have the same
    # voxel size, so this code just loads the first one to determine
    # what it should be
    firstROIFile = op.join(gconf.get_cmp_tracto_mask_tob0(), 
                           gconf.parcellation.keys()[0],
                           'ROI_HR_th.nii.gz')
    firstROI = nibabel.load(firstROIFile)
    roiVoxelSize = firstROI.get_header().get_zooms()
    (endpoints,endpointsmm) = create_endpoints_array(fib, roiVoxelSize)
    np.save(en_fname, endpoints)
    np.save(en_fnamemm, endpointsmm)

    # only compute curvature if required
    if gconf.compute_curvature:
        meancurv = compute_curvature_array(fib)
        np.save(curv_fname, meancurv)
    
    log.info("========================")
    
    n = len(fib)
    
    resolution = gconf.parcellation.keys()

    for r in resolution:
        
        log.info("Resolution = "+r)
        
        # create empty fiber label array
        fiberlabels = np.zeros( (n, 2) )
        final_fiberlabels = []
        final_fibers_idx = []
        
        # Open the corresponding ROI
        log.info("Open the corresponding ROI")
        roi_fname = op.join(gconf.get_cmp_tracto_mask_tob0(), r, 'ROI_HR_th.nii.gz')
        roi       = nibabel.load(roi_fname)
        roiData   = roi.get_data()
      
        # Create the matrix
        nROIs = gconf.parcellation[r]['number_of_regions']
        log.info("Create the connection matrix (%s rois)" % nROIs)
        G     = nx.Graph()

        # add node information from parcellation
        gp = nx.read_graphml(gconf.parcellation[r]['node_information_graphml'])
        for u,d in gp.nodes_iter(data=True):
            G.add_node(int(u), d)

        dis = 0
        
        log.info("Create the connection matrix")
        log.info("Check endpoints closeness to GM:")
        bins = np.arange((gconf.parcellation[r]['number_of_regions']+2))
        added_fibers = np.array([], np.int16)
        zeroep = 0
        mod = 0
        naf = 0
        pc = -1
        for i in range(endpoints.shape[0]):
    
            # ROI start => ROI end
            try:
                startROI = int(roiData[endpoints[i, 0, 0], endpoints[i, 0, 1], endpoints[i, 0, 2]])
                endROI   = int(roiData[endpoints[i, 1, 0], endpoints[i, 1, 1], endpoints[i, 1, 2]])
            except IndexError:
                log.info("An index error occured for fiber %s. This means that the fiber start or endpoint is outside the volume. Continue." % i)
                continue

            # Check for endpoints close to GM
            if startROI == 0 or endROI == 0:
                # If startROI is 0 but close to a non-zero point, correct its value
                if startROI == 0:
                    zeroep += 1
                    # Extract the 3x3x3 voxels neighbourhood of the current point
                    localROI = extract(roiData, shape=(3,3,3), position=(endpoints[i, 0, 0], endpoints[i, 0, 1], endpoints[i, 0, 2]), fill=0)
                    hist, bins_edges = np.histogram(localROI, bins)
                    hist, bins_edges = hist[1:len(hist)], bins[1:(len(bins_edges))]
                    maxindex = hist.argmax()
                    if hist[maxindex] != 0:
                        startROI = int(bins_edges[maxindex])
                        mod += 1
                # If endROI is 0 but close to a non-zero point, correct its value
                if endROI == 0:
                    zeroep += 1
                    # Extract the 3x3x3 voxels neighbourhood of the current point
                    localROI = extract(roiData, shape=(3,3,3), position=(endpoints[i, 1, 0], endpoints[i, 1, 1], endpoints[i, 1, 2]), fill=0)
                    hist, bins_edges = np.histogram(localROI, bins)
                    hist, bins_edges = hist[1:len(hist)], bins[1:(len(bins_edges))]
                    maxindex = hist.argmax()
                    if hist[maxindex] != 0:
                        startROI = int(bins_edges[maxindex])
                        mod += 1
                # If now the fiber is not orphan anymore, memorize it in a vector
                if startROI != 0 and endROI != 0:
                    added_fibers = np.append(added_fibers, i)
                    naf += 1

            # Percent counter
            pcN = int(round( float(100*i)/n ))
            if pcN > pc and pcN%1 == 0:
                pc = pcN
                log.info('%4.0f%%' % (pc))

            # Filter
            if startROI == 0 or endROI == 0:
                dis += 1
                fiberlabels[i,0] = -1
                continue
            
            if startROI > nROIs or endROI > nROIs:
                log.debug("Start or endpoint of fiber terminate in a voxel which is labeled higher")
                log.debug("than is expected by the parcellation node information.")
                log.debug("Start ROI: %i, End ROI: %i" % (startROI, endROI))
                log.debug("This needs bugfixing!")
                continue
            
            # Update fiber label
            # switch the rois in order to enforce startROI < endROI
            if endROI < startROI:
                tmp = startROI
                startROI = endROI
                endROI = tmp

            fiberlabels[i,0] = startROI
            fiberlabels[i,1] = endROI

            final_fiberlabels.append( [ startROI, endROI ] )
            final_fibers_idx.append(i)


            # Add edge to graph
            if G.has_edge(startROI, endROI):
                G.edge[startROI][endROI]['fiblist'].append(i)
            else:
                G.add_edge(startROI, endROI, fiblist   = [i])
                
        log.info("Found %i (%f percent out of %i fibers) fibers that start or terminate in a voxel which is not labeled. (orphans)" % (dis, dis*100.0/n, n) )
        log.info("Valid fibers: %i (%f percent)" % (n-dis, 100 - dis*100.0/n) )
        log.info("Modified %i (%f percent out of %i endpoints in the WM) endpoints in the WM but close to the GM" % (mod, mod*100.0/zeroep, zeroep) )
        log.info("Valid fibers after endpoints extension: %i (%f percent)" % (n-dis+naf, (n-dis+naf)*100.0/n) )

        # create a final fiber length array
        finalfiberlength = []
        for idx in final_fibers_idx:
            # compute length of fiber
            finalfiberlength.append( length(fib[idx][0]) )

        # convert to array
        final_fiberlength_array = np.array( finalfiberlength )
        
        # make final fiber labels as array
        final_fiberlabels_array = np.array(final_fiberlabels, dtype = np.int32)

        # update edges
        # measures to add here
        for u,v,d in G.edges_iter(data=True):
            G.remove_edge(u,v)
            di = { 'number_of_fibers' : len(d['fiblist']), }
            
            # additional measures
            # compute mean/std of fiber measure
            idx = np.where( (final_fiberlabels_array[:,0] == int(u)) & (final_fiberlabels_array[:,1] == int(v)) )[0]

            di['fiber_length_mean'] = np.mean(final_fiberlength_array[idx])
            di['fiber_length_std'] = np.std(final_fiberlength_array[idx])

            G.add_edge(u,v, di)

        # storing network
        nx.write_gpickle(G, op.join(gconf.get_cmp_matrices(), 'connectome_%s.gpickle' % r))

        log.info("Storing final fiber length array")
        fiberlabels_fname  = op.join(gconf.get_cmp_fibers(), 'final_fiberslength_%s.npy' % str(r))
        np.save(fiberlabels_fname, final_fiberlength_array)

        log.info("Storing all fiber labels (with orphans)")
        fiberlabels_fname  = op.join(gconf.get_cmp_fibers(), 'filtered_fiberslabel_%s.npy' % str(r))
        np.save(fiberlabels_fname, np.array(fiberlabels, dtype = np.int32), )

        log.info("Storing final fiber labels (no orphans)")
        fiberlabels_noorphans_fname  = op.join(gconf.get_cmp_fibers(), 'final_fiberlabels_%s.npy' % str(r))
        np.save(fiberlabels_noorphans_fname, final_fiberlabels_array)

        log.info("Filtering tractography - keeping only no orphan fibers")
        finalfibers_fname = op.join(gconf.get_cmp_fibers(), 'streamline_final_%s.trk' % str(r))
        save_fibers(hdr, fib, finalfibers_fname, final_fibers_idx)

    log.info("Done.")
    log.info("========================")

def inspect(gconf):
    """ Inspect the results of this stage """
    log = gconf.get_logger()

    # display matrices with matplotlib if available
    # the same script as used in connectome viewer

    # Importing NetworkX
    import networkx as nx

    # Import Pylab
    try:
        from cmp.util import show_matrix
    except ImportError:
        log.info("matplotlib not available. Can not plot matrix")
        return

    resolution = gconf.parcellation.keys()

    for r in resolution:
        # retrieve the graph
        g=nx.read_gpickle(op.join(gconf.get_cmp_matrices(), 'connectome_%s.gpickle' % r))
        show_matrix(g, "number_of_fibers", True)


def run(conf):
    """ Run the connection matrix module
    
    Parameters
    ----------
    conf : PipelineConfiguration object
    
    """
    # setting the global configuration variable
    globals()['gconf'] = conf
    globals()['log'] = gconf.get_logger() 
    start = time()

    log.info("Connectome Matrix Creation")
    log.info("==========================")

    cmat()
            
    log.info("Module took %s seconds to process." % (time()-start))
    
    if not len(gconf.emailnotify) == 0:
        msg = ["Create connectome", int(time()-start)]
        send_email_notification(msg, gconf, log)  
        

def declare_inputs(conf):
    """Declare the inputs to the stage to the PipelineStatus object"""
    
    stage = conf.pipeline_status.GetStage(__name__)
    
    conf.pipeline_status.AddStageInput(stage, conf.get_cmp_fibers(), 'streamline_filtered.trk', 'streamline-trk')
    
    for r in conf.parcellation.keys():
        conf.pipeline_status.AddStageInput(stage, op.join(conf.get_cmp_tracto_mask_tob0(), r), 'ROI_HR_th.nii.gz', 'ROI_HR_th_%s-nii-gz' % r)
        
def declare_outputs(conf):
    """Declare the outputs to the stage to the PipelineStatus object"""
    
    stage = conf.pipeline_status.GetStage(__name__)
            
    conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_fibers(), 'endpoints.npy', 'endpoints-npy')       
    conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_fibers(), 'endpointsmm.npy', 'endpointsmm-npy')
    
    
    resolution = conf.parcellation.keys()
    for r in resolution:
        conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_matrices(), 'connectome_%s.gpickle' % r, 'connectome_%s-gpickle')
        conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_fibers(), 'final_fiberslength_%s.npy' % str(r), 'final_fiberslength_%s-npy' % str(r))
        conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_fibers(), 'filtered_fiberslabel_%s.npy' % str(r), 'filtered_fiberslabel_%s-npy' % str(r))
        conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_fibers(), 'final_fiberlabels_%s.npy' % str(r), 'fiberlabels_%s-npy' % str(r))
        conf.pipeline_status.AddStageOutput(stage, conf.get_cmp_fibers(), 'streamline_final_%s.trk' % str(r), 'streamline_final_%s-trk' % str(r))
        
        