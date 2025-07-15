import sys
sys.path.insert(1, '../../eispy2d/library/')

import inputdata as ipt
import casestudy as cst
import richmond as ric
import regularization as reg
import mom_cg_fft as mom
import stopcriteria as stp
import lsm
import osm
import bim
import csi
import som

name = 'star'
resolution = (40, 40)

test = ipt.InputData(import_filename=name + '.ipt')

method = [
    lsm.LinearSamplingMethod(alias='lsm',
                             regularization=reg.ConjugatedGradient(300),
                             sv_cutoff=None, threshold=.7),
    osm.OrthogonalitySamplingMethod(threshold=.35),
    bim.BornIterativeMethod(mom.MoM_CG_FFT(), reg.ConjugatedGradient(300), 
                            stp.StopCriteria(max_iterations=30)),
    csi.ContrastSourceInversion(stp.StopCriteria(max_iterations=300)),
    som.SubspaceBasedOptimizationMethod(stp.StopCriteria(max_iterations=30),
                                        cutoff_index=5)
]

discretization = ric.Richmond(configuration=test.configuration,
                              elements=resolution)

casestudy = cst.CaseStudy(name=name + '.cst',
                           method=method,
                           discretization=discretization,
                           test=test)

casestudy.run(parallelization=cst.PARALLELIZE_METHOD, pre_save=True)

casestudy.save(save_test=True)