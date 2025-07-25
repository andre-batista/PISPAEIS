from matplotlib import pyplot as plt
import sys
sys.path.insert(1, '../../eispy2d/library/')
import casestudy as cst
import result as rst

# Configuration
plt.rcParams['image.cmap'] = 'Greys'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

def save_figure(filepath, filename, tight_layout=True):
    """Save figure with consistent settings."""
    if tight_layout:
        plt.tight_layout()
    plt.savefig(f'{filepath}{filename}.eps', format='eps', bbox_inches='tight')
    plt.close()

def configure_axes(axis, hide_x_labels=True, hide_y_labels=True):
    """Configure axis labels consistently."""
    if hide_x_labels:
        axis[0].set_xticklabels([])
    if hide_y_labels:
        axis[0].set_yticklabels([])

# Initialize case study
mycase = cst.CaseStudy(import_filename='breastphantom.cst', 
                       import_filepath='../../data/breast/class2/phantom1/')
filepath = './figs/'

# Generate ground truth figure
fig, axis, _ = rst.get_figure(1)
mycase.test.draw(fontsize=20, axis=axis, title=False)
# axis[0].set_xticks([-1, 0, 1])
configure_axes(axis, hide_y_labels=False)
save_figure(filepath, '0')

# Methods to generate reconstructions
methods = [
    ('lsm', '1', True, True),
    ('osm', '2', True, True),
    ('bim', '3', False, False), 
    ('csi', '4', False, True),  # Keep original labels for CSI
    ('som', '5', False, True),
    ('ca',  '6', False, True)
]

# Generate reconstruction figures
for method, filename, hide_x, hide_y in methods:
    axis = mycase.reconstruction(fontsize=20, method=method, title=False)
    configure_axes(axis, hide_x, hide_y)
    save_figure(filepath, filename)