import sys
sys.path.insert(1, '../../../eispy2d/library/')

import inputdata as ipt
import som
import stopcriteria as stp
import richmond as ric

file_path = '../../../data/position/single/'
file_name = 'star4.ipt'

test = ipt.InputData.importdata(import_filename=file_name,
                                import_filepath=file_path)

resolution = (40, 40)

discretization = ric.Richmond(configuration=test.configuration,
                              elements=resolution)

method = som.SubspaceBasedOptimizationMethod(
    stp.StopCriteria(max_iterations=600), cutoff_index=15
)

result = method.solve(test, discretization, print_info=True)

result.save(file_path=file_path)
