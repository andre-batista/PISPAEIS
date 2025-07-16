import sys
sys.path.insert(1, '../../eispy2d/library/')
import configuration as cfg
import inputdata as ipt
import result as rst
import draw
import mom_cg_fft as mom

name = 'vary.cfg'
NM = NS = 60
Ro = 4.
lambda_b = 1.
epsilon_rb = 1.
Lx = Ly = 2.
E0 = 1.
perfect_dielectric = True

config = cfg.Configuration(name=name, number_measurements=NM, number_sources=NS,
                           observation_radius=Ro, wavelength=lambda_b,
                           background_permittivity=epsilon_rb,
                           image_size=[Ly, Lx], magnitude=E0,
                           perfect_dielectric=perfect_dielectric)

epsilon_rd = [1.25, 1.5, 1.75, 2., 2.5]
name = 'vary'
resolution = (160, 160)
noise = 5.
indicators = [rst.SHAPE_ERROR, rst.POSITION_ERROR]

forward = mom.MoM_CG_FFT(tolerance=1e-5, maximum_iterations=5000,
                         parallelization=True)

for n in range(len(epsilon_rd)):

    print('Test %d... ' % n, end='')
    test = ipt.InputData(name=name + '%d' % n + '.ipt', configuration=config,
                         resolution=resolution, noise=noise,
                         indicators=indicators)

    chi = (epsilon_rd[n]-epsilon_rb)/epsilon_rb
    l = 0.9
    position = [.0, .0]

    test.rel_permittivity, _ = draw.star5(
        l, axis_length_x=Lx, axis_length_y=Ly, resolution=resolution,
        background_rel_permittivity=epsilon_rb, 
        object_rel_permittivity=epsilon_rd[n], center=position, rotate=0.
    )
    
    print('solving forward problem... ', end='')
    _ = forward.solve(test)
    
    print('computing DNL... ', end='')
    test.compute_dnl()
    
    print('ok!')
    test.save()
