import sys
sys.path.insert(1, '../../../../eispy2d/library/')

import inputdata as ipt
import casestudy as cst
import richmond as ric
import circleapproximation as ca
import stochastic as stc

name = 'single'
resolution = (40, 40)

test = ipt.InputData(import_filename=name + '.ipt',
                     import_filepath='./data/')

contrast_range = 10.
number_executions = 30

method = ca.CircleApproximation(stc.OutputMode(stc.EACH_EXECUTION),
                                contrast_range=contrast_range)

discretization = ric.Richmond(configuration=test.configuration,
                              elements=resolution)

casestudy = cst.CaseStudy(name=name + '.cst',
                          stochastic_runs=number_executions,
                          save_stochastic_runs=True,
                           method=method,
                           discretization=discretization,
                           test=test)

casestudy.run(parallelization=False, pre_save=True)

casestudy.save(file_path='./data/', save_test=True)