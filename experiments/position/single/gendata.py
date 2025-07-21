import sys
sys.path.insert(1, '../../../eispy2d/library/')

import configuration as cfg
import inputdata as ipt
import result as rst
import draw
import mom_cg_fft as mom

# Parameters
name = 'single'
file_path = "../../../data/position/single/"
NM = NS = 80
Ro = 4.
lambda_b = 1.
epsilon_rb = 1.
Lx = Ly = 2.
E0 = 1.
perfect_dielectric = True
resolution = (160, 160)
noise = 5.
indicators = [rst.SHAPE_ERROR, rst.POSITION_ERROR, rst.OBJECTIVE_FUNCTION]
epsilon_rd = 6.
l = 0.3
position = [.4, -.4]
rotate = 30

config = cfg.Configuration(name=name + '.cfg', number_measurements=NM, 
                           number_sources=NS, observation_radius=Ro, 
                           wavelength=lambda_b,
                           background_permittivity=epsilon_rb,
                           image_size=[Ly, Lx], magnitude=E0,
                           perfect_dielectric=perfect_dielectric)


test = ipt.InputData(name=name + '.ipt', configuration=config, 
                     resolution=resolution, noise=noise, indicators=indicators)

chi = (epsilon_rd-epsilon_rb)/epsilon_rb

test.rel_permittivity, _ = draw.star5(
    l, axis_length_x=Lx, axis_length_y=Ly, resolution=resolution,
    background_rel_permittivity=epsilon_rb, object_rel_permittivity=epsilon_rd,
    center=position, rotate=30.
)

test.draw(show=True)

forward = mom.MoM_CG_FFT(tolerance=1e-3, maximum_iterations=10000,
                         parallelization=True)

_ = forward.solve(test, PRINT_INFO=False)

test.compute_dnl()

test.save(file_path=file_path)