import sys
sys.path.insert(1, '../../../eispy2d/library/')
import casestudy as cst
import numpy as np
from matplotlib import pyplot as plt
import result as rst
import configuration as cfg

casestudy = cst.CaseStudy(
    import_filename='star.cst',
    import_filepath='../../../data/indicatorcomparison/shape/'
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 40
plt.rcParams['image.cmap'] = 'Greys'


filepath = './figs/'
config = casestudy.test.configuration
xlabel, ylabel = r'x [$\lambda_b$]', r'y [$\lambda_b$]'
clb_contrast = r'$|\chi|$'
xmin, xmax = cfg.get_bounds(config.Lx)
ymin, ymax = cfg.get_bounds(config.Ly)
extent = [xmin/config.lambda_b,
          xmax/config.lambda_b,
          ymin/config.lambda_b,
          ymax/config.lambda_b]

result = casestudy.test
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=config)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=0.3)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
# plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
# plt.show()
plt.tight_layout(h_pad=0.6, w_pad=0.6)
plt.savefig(filepath + 'groundtruth.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudy.results[0]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=config)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=0.3)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([])
plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
# plt.show()
plt.tight_layout(h_pad=0.6, w_pad=0.6)
plt.savefig(filepath + 'som10.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudy.results[1]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=0.3)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([])
plt.xlabel(xlabel)
# plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som20.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudy.results[2]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=0.3)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([])
plt.xlabel(xlabel)
# plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som30.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudy.results[3]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(12, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=0.3)
plt.colorbar(label=clb_contrast, shrink=.9)
plt.xticks([-1, 0, 1])
plt.yticks([])
plt.xlabel(xlabel)
# plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som100.eps', format='eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 10))
plt.plot(casestudy.results[-1].ssim, '-k', linewidth=10, markersize=30)
plt.xlabel('Iterations', fontsize=30)
plt.ylabel(rst.LABELS[rst.SSIM_ERROR], fontsize=30)
plt.grid()
plt.tight_layout()
plt.savefig(filepath + 'ssim.eps', format='eps', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.plot(casestudy.results[-1].zeta_epad, '-k', linewidth=10, 
         markersize=30)
plt.xlabel('Iterations', fontsize=30)
plt.ylabel(rst.LABELS[rst.REL_PERMITTIVITY_PAD_ERROR], fontsize=30)
plt.grid()
plt.tight_layout()
plt.savefig(filepath + 'zeta_epad.eps', format='eps', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure(figsize=(10, 10))
plt.plot(casestudy.results[-1].zeta_s, '-k', linewidth=10, 
         markersize=30)
plt.xlabel('Iterations', fontsize=30)
plt.ylabel(rst.LABELS[rst.SHAPE_ERROR], fontsize=30)
plt.grid()
plt.tight_layout()
plt.savefig(filepath + 'zeta_s.eps', format='eps', bbox_inches='tight')
# plt.show()
plt.close()