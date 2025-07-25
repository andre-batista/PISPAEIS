from loadphantom import BreastPhantomReader
file_path = "../../data/breast/class2/phantom1/"
reader = BreastPhantomReader(file_path)

freq = 1. # GHz
epsilon_rb = 10.
sigma_b = 0.
noise = 5.

reader.set_immersion_medium_properties(dielectric_constant=epsilon_rb,
                                       conductivity=sigma_b)

epsilon_r, sigma = reader.calculate_dielectric_properties(
    freq, model='debye'
)

import numpy as np
import sys
sys.path.insert(1, '../../eispy2d/library/')
import configuration as cfg

epsr = np.squeeze(epsilon_r[140, :, :])
sig = np.squeeze(sigma[140, :, :])

# sig = None

instance_name = 'breastphantom'
dx = dy = 0.5e-3 # 0.5 mm
Lx, Ly = epsr.shape[1] * dx, epsr.shape[0] * dy
max_epsilon_r = epsr.max()
object_radius = min([Lx, Ly]) / 2.0
DOF = cfg.degrees_of_freedom(object_radius, frequency=freq*1e9,
                             epsilon_r=max_epsilon_r)
print(f"Degrees of freedom: {DOF}")
NS = NM = 80
Ro = max([Lx, Ly])
image_size = [Ly, Lx]
E0 = 10.

config = cfg.Configuration(name=instance_name + '.cfg',
                           number_measurements=NM,
                           number_sources=NS,
                           observation_radius=Ro,
                           frequency=freq*1e9,
                           background_permittivity=epsilon_rb,
                           background_conductivity=sigma_b,
                           image_size=image_size,
                           wavelength_unit=False,
                           magnitude=E0,
                           perfect_dielectric=False)
import inputdata as ipt
import result as rst

test = ipt.InputData(name=instance_name + '.ipt',
                     configuration=config,
                     rel_permittivity=epsr,
                     conductivity=sig,
                     indicators=[rst.SHAPE_ERROR, rst.POSITION_ERROR,
                                 rst.OBJECTIVE_FUNCTION])
test.compute_dnl()

import mom_cg_fft as mom
forward = mom.MoM_CG_FFT(tolerance=1e-3, maximum_iterations=5_000)
_ = forward.solve(test, noise=noise)

test.save(file_path=file_path)