import sys
sys.path.insert(1, '../../../eispy2d/library/')

import testset as tst
import richmond as ric
import regularization as reg
import mom_cg_fft as mom
import stopcriteria as stp
import lsm
import osm
import bim
import csi
import som
import benchmark as bmk
import circleapproximation as ca
import stochastic as stc

name = 'average'
filepath = '../../../data/position/average/'
resolution = (40, 40)
contrast_range = (0.1, 10.)

testset = tst.TestSet(import_filename=name + '.tst',
                      import_filepath=filepath)

method = [
    lsm.LinearSamplingMethod(alias='lsm',
                             regularization=reg.ConjugatedGradient(300),
                             sv_cutoff=None, threshold=.7),
    osm.OrthogonalitySamplingMethod(threshold=.35),
    bim.BornIterativeMethod(mom.MoM_CG_FFT(), reg.ConjugatedGradient(300), 
                            stp.StopCriteria(max_iterations=30)),
    csi.ContrastSourceInversion(stp.StopCriteria(max_iterations=300)),
    som.SubspaceBasedOptimizationMethod(stp.StopCriteria(max_iterations=30),
                                        cutoff_index=5),
    ca.CircleApproximation(stc.OutputMode(stc.BEST_CASE, reference='zeta_s'),
                           number_executions=1,
                           contrast_range=contrast_range,
                           solver="de")
]

discretization = ric.Richmond(configuration=testset.test[0].configuration,
                              elements=resolution)

benchmark = bmk.Benchmark(name=name + '.bmk', method=method, 
                          discretization=discretization, testset=testset)

benchmark.run(parallelization=bmk.PARALLELIZE_EXECUTIONS, pre_save=True,
              file_path=filepath)

benchmark.save(file_path=filepath, save_testset=True)