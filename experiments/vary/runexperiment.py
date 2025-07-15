import sys
sys.path.insert(1, '../../eispy2d/library/')

import casestudy as cst
import richmond as ric
import inputdata as ipt
import osm

name = 'vary'
resolution = (40, 40)

tests = []
for n in range(5):
    tests.append(ipt.InputData(import_filename=name + '%d' % n + '.ipt'))

method = osm.OrthogonalitySamplingMethod(threshold=.35)

discretization = ric.Richmond(configuration=tests[0].configuration,
                              elements=resolution)

for n in range(len(tests)):

    caseestudy = cst.CaseStudy(name=name + '%d' % n + '.cst', method=method,
                               discretization=discretization, test=tests[n])

    caseestudy.run(parallelization=cst.PARALLELIZE_METHOD, pre_save=True)

    caseestudy.save(save_test=True)