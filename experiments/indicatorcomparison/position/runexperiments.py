import sys
sys.path.insert(1, '../../../eispy2d/library/')

import inputdata as ipt
import casestudy as cst
import richmond as ric
import regularization as reg
import mom_cg_fft as mom
import stopcriteria as stp
import stochastic as stc
import circleapproximation as ca

name = 'star'
filepath = '../../../data/indicatorcomparison/position/'
resolution = (40, 40)

test = ipt.InputData(import_filename=name + '.ipt',
                     import_filepath=filepath)

contrast_range = (0., 10.)

method = [
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
                           test=test)

casestudy.run(parallelization=cst.PARALLELIZE_METHOD, pre_save=True,
              file_path=filepath)

casestudy.save(save_test=True, file_path=filepath)