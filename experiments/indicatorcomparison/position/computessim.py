import sys
sys.path.insert(1, '../../../eispy2d/library/')
import casestudy as cst
import inputdata as ipt
import result as rst
import configuration as cfg

groundtruth = ipt.InputData(
     import_filename='star.ipt',
     import_filepath='../../../data/indicatorcomparison/position/'
)
casestudy = cst.CaseStudy(
    import_filename='star.cst',
    import_filepath='../../../data/indicatorcomparison/position/'
)

if casestudy.results[0].ssim is None or len(casestudy.results[0].ssim) == 0:

    Xo = cfg.get_contrast_map(epsilon_r=groundtruth.rel_permittivity,
                                        sigma=groundtruth.conductivity,
                                        configuration=groundtruth.configuration)
    result = casestudy.results[0]

    Xr = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                              sigma=result.conductivity,
                              configuration=result.configuration)

    ssim = rst.compute_ssim(Xo, Xr)

    casestudy.results[0].ssim = [ssim]

    casestudy.save(file_path='../../../data/indicatorcomparison/position/')

else:
    print("SSIM data already available.")