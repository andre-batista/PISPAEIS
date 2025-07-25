import sys
sys.path.insert(1, '../../../eispy2d/library/')
import casestudy as cst
import numpy as np
from matplotlib import pyplot as plt
import result as rst
import configuration as cfg

N = 5
casestudies = [cst.CaseStudy(import_filename='vary%d.cst' % n,
                             import_filepath='../../../data/shape/vary/') for n in range(N)]
chi = np.array([.25, .5, .75, 1., 1.5])

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 40

filepath = './figs/'
config = casestudies[0].test.configuration
xlabel, ylabel = r'x [$\lambda_b$]', r'y [$\lambda_b$]'
clb_contrast = r'$|\chi|$'
xmin, xmax = cfg.get_bounds(config.Lx)
ymin, ymax = cfg.get_bounds(config.Ly)
extent = [xmin/config.lambda_b,
          xmax/config.lambda_b,
          ymin/config.lambda_b,
          ymax/config.lambda_b]

result = casestudies[0].results[0]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=config)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=1)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
# plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
# plt.show()
plt.tight_layout(h_pad=0.6, w_pad=0.6)
plt.savefig(filepath + 'osm_0.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[1].results[0]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=1)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'osm_1.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[2].results[0]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=1)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'osm_2.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[3].results[0]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=1)
# plt.colorbar(label=clb_contrast)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'osm_3.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[4].results[0]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal', vmin=0, vmax=1)
plt.colorbar(label=clb_contrast, shrink=.6)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'osm_4.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[0].results[1]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal')
plt.colorbar(label=clb_contrast, shrink=.7)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som_0.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[1].results[1]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal')
plt.colorbar(label=clb_contrast, shrink=.7)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som_1.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[2].results[1]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal')
plt.colorbar(label=clb_contrast, shrink=.7)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som_2.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[3].results[1]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal')
plt.colorbar(label=clb_contrast, shrink=.7)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout(h_pad=0.6, w_pad=0.6)
# plt.show()
plt.savefig(filepath + 'som_3.eps', format='eps', bbox_inches='tight')
plt.close()

result = casestudies[4].results[1]
X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                         sigma=result.conductivity,
                         configuration=result.configuration)
X = np.abs(X)
plt.figure(figsize=(10, 10))
plt.imshow(X, origin='lower', extent=extent, cmap='Greys',
           aspect='equal')
plt.colorbar(label=clb_contrast, shrink=.7)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.tight_layout()
# plt.show()
plt.savefig(filepath + 'som_4.eps', format='eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 10))
error = np.zeros((N, 2))
for n in range(N):
    error[n,0] = casestudies[n].results[0].zeta_s[-1]
    error[n,1] = casestudies[n].results[1].zeta_s[-1]
plt.plot(chi, error[:, 0], 'o-k', linewidth=10, markersize=30, label='OSM')
plt.plot(chi, error[:, 1], '*:k', linewidth=10, markersize=30, label='SOM')
plt.xlabel(r'$\chi$')
plt.ylabel(r'$\zeta_s$ [%]')
plt.legend(loc='upper left', fontsize=30)
plt.grid()
plt.tight_layout()
plt.savefig(filepath + 'zeta_s.eps', format='eps', bbox_inches='tight')
# plt.show()
plt.close()

dnl = np.zeros(N)
for n in range(N):
    dnl[n] = casestudies[n].test.dnl
plt.figure(figsize=(10, 10))
plt.plot(chi, dnl, 'o--k', linewidth=10, markersize=30)
plt.xlabel(r'$\chi$')
plt.ylabel('DNL')
plt.grid()
plt.tight_layout()
plt.savefig(filepath + 'dnl.eps', format='eps', bbox_inches='tight')
plt.close()