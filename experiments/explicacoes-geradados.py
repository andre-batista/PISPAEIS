import sys
sys.path.insert(1, '../../eispy2d/library/')
import configuration as cfg
import inputdata as ipt
import result as rst
import draw
import mom_cg_fft as mom
import osm
import regularization as reg
import richmond as ric

name = 'test.cfg'
NM = NS = 40
Ro = 3.
lambda_b = 1.
epsilon_rb = 1.
Lx = Ly = 4.
E0 = 1.
perfect_dielectric = True

config = cfg.Configuration(name=name, number_measurements=NM, number_sources=NS,
                           observation_radius=Ro, wavelength=lambda_b,
                           background_permittivity=epsilon_rb,
                           image_size=[Ly, Lx], magnitude=E0,
                           perfect_dielectric=perfect_dielectric)


name = 'test.ipt'
resolution = (150, 150)
noise = 5.
indicators = [rst.SHAPE_ERROR, rst.POSITION_ERROR]

test = ipt.InputData(name=name, configuration=config, resolution=resolution, 
                     noise=noise, indicators=indicators)

epsilon_rd = 1.25
chi = (epsilon_rd-epsilon_rb)/epsilon_rb
l = 0.9
position = [.4, .4]

test.rel_permittivity, _ = draw.square(
    l, axis_length_x=Lx, axis_length_y=Ly, resolution=resolution,
    background_rel_permittivity=epsilon_rb, object_rel_permittivity=epsilon_rd,
    center=position, rotate=0.
)

position = [-1, -1]

test.rel_permittivity, _ = draw.triangle(
    l, axis_length_x=Lx, axis_length_y=Ly,
    background_rel_permittivity=epsilon_rb, object_rel_permittivity=epsilon_rd,
    center=position, rotate=0., rel_permittivity=test.rel_permittivity
)

forward = mom.MoM_CG_FFT(tolerance=1e-3, maximum_iterations=5000,
                         parallelization=True)

_ = forward.solve(test)

threshold = 0.35

method = osm.OrthogonalitySamplingMethod(threshold=threshold)

image = (50, 50)

discretization = ric.Richmond(configuration=config, elements=image)

result = method.solve(test, discretization=discretization)

test.save(file_path='./data/')
result.name = 'test-osm.rst'
result.save(file_path='./data/')