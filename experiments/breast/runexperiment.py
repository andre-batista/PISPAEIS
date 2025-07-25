import sys
sys.path.insert(1, '../../eispy2d/library/')

import inputdata as ipt
import casestudy as cst
import richmond as ric
import regularization as reg
import mom_cg_fft as mom
import stopcriteria as stp
import stochastic as stc
import lsm
import osm
import bim
import csi
import som
import circleapproximation as ca

name = 'breastphantom'
file_path = "../../data/breast/"
resolution = (40, 40)
contrast_range = (0.1, 10.)

test = ipt.InputData(import_filename=name + '.ipt', import_filepath=file_path)

method = [
    lsm.LinearSamplingMethod(alias='lsm',
                             regularization=reg.ConjugatedGradient(300),
                             sv_cutoff=None, threshold=.7),
    osm.OrthogonalitySamplingMethod(threshold=.35),
    bim.BornIterativeMethod(mom.MoM_CG_FFT(), reg.ConjugatedGradient(300), 
                            stp.StopCriteria(max_iterations=30)),
    csi.ContrastSourceInversion(stp.StopCriteria(max_iterations=1_000)),
    som.SubspaceBasedOptimizationMethod(stp.StopCriteria(max_iterations=30),
                                        cutoff_index=15),
    ca.CircleApproximation(stc.OutputMode(stc.BEST_CASE, reference='zeta_s'),
                           number_executions=1,
                           contrast_range=contrast_range,
                           solver="de")
]

discretization = ric.Richmond(configuration=test.configuration,
                              elements=resolution)

casestudy = cst.CaseStudy(name=name + '.cst',
                           method=method,
                           discretization=discretization,
                           test=test, stochastic_runs=1)

casestudy.run(parallelization=cst.PARALLELIZE_METHOD, pre_save=True,
              file_path=file_path)

casestudy.save(save_test=True, file_path=file_path)