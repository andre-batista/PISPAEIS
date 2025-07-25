import sys
sys.path.insert(1, '../../../eispy2d/library/')
import casestudy as cst
from matplotlib import pyplot as plt
import result as rst
import inputdata as ipt

test = ipt.InputData(import_filename='star.ipt',
                     import_filepath='../../../data/shape/star/')

casestudy = cst.CaseStudy(import_filename='star.cst',
                          import_filepath='../../../data/shape/star/')
casestudy.test = test

filepath = './figs/'
plt.rcParams['image.cmap'] = 'Greys'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
fig, axis, _ = rst.get_figure(1)
test.draw(fontsize=20, axis=axis, title=False, vmax=0.35)
# axis[0].set_xticks([-1, 0, 1])
axis[0].set_xticklabels([])
# # Ajustar os ticks da colorbar existente
# if fig.axes[-1].images:  # Verifica se há uma imagem associada à última barra
#     cbar = fig.axes[-1]
#     cbar.set_ticks([0.1, 0.2, 0.3])  #
plt.tight_layout()
plt.savefig(filepath + '0.eps', format='eps')
plt.close()
axis = casestudy.reconstruction(fontsize=20, method='lsm', title=False)
axis[0].set_xticklabels([])
axis[0].set_yticklabels([])
plt.tight_layout()
plt.savefig(filepath + '1.eps', format='eps')
plt.close()
axis = casestudy.reconstruction(fontsize=20, method='osm', title=False)
axis[0].set_xticklabels([])
axis[0].set_yticklabels([])
plt.tight_layout()
plt.savefig(filepath + '2.eps', format='eps')
plt.close()
axis = casestudy.reconstruction(fontsize=20, method='bim', vmax=0.35, title=False)
# axis[0].set_xticklabels([])
# axis[0].set_yticklabels([])
plt.tight_layout()
plt.savefig(filepath + '3.eps', format='eps')
plt.close()
axis = casestudy.reconstruction(fontsize=20, method='csi', vmax=0.35, title=False)
# axis[0].set_xticklabels([])
axis[0].set_yticklabels([])
plt.tight_layout()
plt.savefig(filepath + '4.eps', format='eps')
plt.close()
axis = casestudy.reconstruction(fontsize=20, method='som', vmax=0.35, title=False)
# axis[0].set_xticklabels([])
axis[0].set_yticklabels([])
plt.tight_layout()
plt.savefig(filepath + '5.eps', format='eps')
plt.close()
# axis = casestudy.reconstruction(fontsize=20, show=True, include_true=True)
# axis[3].images[0].set_clim(vmax=0.35)
# axis[4].images[0].set_clim(vmax=0.35)
# axis[5].images[0].set_clim(vmax=0.35)
# plt.tight_layout(h_pad=0.6, w_pad=0.6)