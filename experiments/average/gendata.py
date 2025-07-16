import sys
sys.path.insert(1, '../../eispy2d/library/')

import configuration as cfg
import testset as tst
from experiment import RANDOM_POLYGONS_PATTERN
import result as rst
import mom_cg_fft as mom

name = 'average'
NM = NS = 20
Ro = 4.
lambda_b = 1.
epsilon_rb = 1.
Lx = Ly = 2.
E0 = 1.
perfect_dielectric = True

config = cfg.Configuration(name=name + '.cfg', number_measurements=NM,
                           number_sources=NS, observation_radius=Ro,
                           wavelength=lambda_b,
                           background_permittivity=epsilon_rb,
                           image_size=[Ly, Lx], magnitude=E0,
                           perfect_dielectric=perfect_dielectric)

contrast = .25
object_size = .9
resolution = (80, 80)
map_pattern = RANDOM_POLYGONS_PATTERN
sample_size = 30
noise = 5.
indicators = [rst.SHAPE_ERROR, rst.POSITION_ERROR]

testset = tst.TestSet(name=name + '.tst', configuration=config,
                      contrast=contrast, object_size=object_size, 
                      resolution=resolution, map_pattern=map_pattern,
                      sample_size=sample_size, noise=noise,
                      indicators=indicators)

testset.randomize_tests(compute_dnl=False)

testset.generate_field_data(solver=mom.MoM_CG_FFT(tolerance=1e-3,
                                                  maximum_iterations=5000))

testset.save(file_path='./data/')