import sys
sys.path.insert(1, '../../eispy2d/library/')
import benchmark as bmk
import result as rst
import numpy as np
import matplotlib.pyplot as plt

# Configuration constants
FIGURE_SIZE = (12, 8)
BOXPLOT_SIZE = (5, 6)
FONT_FAMILY = 'Times New Roman'
FONT_SIZES = {'labels': 30, 'legend': 20, 'ticks': 25, 'boxplot': 25}
LINE_STYLE = {'markersize': 11, 'linewidth': 3}
OUTPUT_CONFIG = {'dpi': 300, 'bbox_inches': 'tight', 'format': 'eps'}

# Visual styling arrays
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
LINE_STYLES = ['-', '--', '-.', ':', '-']
GRAY_COLORS = ['0.7', '0.6', '0.4', '0.2', '0.0']  # light to dark gray

# Algorithm names for legend
ALGORITHM_NAMES = ['LSM', 'OSM', 'BIM', 'CSI', 'SOM']

def setup_matplotlib_defaults():
    """Set global matplotlib configuration."""
    plt.rcParams['font.family'] = FONT_FAMILY

def setup_line_styles(axis):
    """Apply visual styling to all lines in the plot."""
    for i, line in enumerate(axis.lines):
        line.set_color(GRAY_COLORS[i % len(GRAY_COLORS)])
        line.set_marker(MARKERS[i % len(MARKERS)])
        line.set_linestyle(LINE_STYLES[i % len(LINE_STYLES)])
        line.set_markersize(LINE_STYLE['markersize'])
        line.set_linewidth(LINE_STYLE['linewidth'])

def setup_font_styling(axis):
    """Apply font styling to axis labels and ticks."""
    axis.set_xlabel(axis.get_xlabel(), fontfamily=FONT_FAMILY, 
                    fontsize=FONT_SIZES['labels'])
    axis.set_ylabel(axis.get_ylabel(), fontfamily=FONT_FAMILY, 
                    fontsize=FONT_SIZES['labels'])
    axis.tick_params(labelsize=FONT_SIZES['ticks'])
    
    # Set font family for tick labels
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

def add_legend_if_needed(axis):
    """Add legend if there are lines to label."""
    if axis.lines and len(axis.lines) <= len(ALGORITHM_NAMES):
        axis.legend(ALGORITHM_NAMES[:len(axis.lines)],
                   loc='upper right', 
                   prop={'family': FONT_FAMILY, 'size': FONT_SIZES['legend']})

def create_zeta_plot(benchmark):
    """Create and style the zeta_s convergence plot."""
    fig, axis, _ = rst.get_figure()
    axis = benchmark.plot("zeta_s", axis=axis[0], show=False)
    
    # Apply all styling
    setup_line_styles(axis)
    setup_font_styling(axis)
    add_legend_if_needed(axis)
    
    # Final figure configuration
    fig.set_size_inches(*FIGURE_SIZE)
    fig.tight_layout()
    fig.savefig('./figs/plot.eps', **OUTPUT_CONFIG)
    
    return fig

def create_dnl_boxplot(benchmark):
    """Create and style the DNL boxplot."""
    # Extract DNL data efficiently
    dnl_data = np.array([test.dnl for test in benchmark.testset.test[:30]])
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=BOXPLOT_SIZE)
    box_plot = ax.boxplot(dnl_data, vert=True, patch_artist=True)
    
    # Style boxplot
    box_plot['boxes'][0].set_facecolor('lightgray')
    box_plot['medians'][0].set_color('black')
    
    # Configure axes
    ax.set_ylabel('DNL', fontsize=FONT_SIZES['boxplot'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['boxplot'])
    ax.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels
    ax.grid(True)
    
    # Save and cleanup
    fig.tight_layout()
    fig.savefig('./figs/dnl.eps', **OUTPUT_CONFIG)
    plt.close(fig)
    
    return fig

# Main execution
if __name__ == "__main__":
    # Setup global configuration
    setup_matplotlib_defaults()
    
    # Load benchmark data
    benchmark = bmk.Benchmark(import_filename="average.bmk",
                             import_filepath="../../data/shape/average/")
    
    # Create plots
    zeta_fig = create_zeta_plot(benchmark)
    dnl_fig = create_dnl_boxplot(benchmark)
    
    print("Plots generated successfully:")
    print("- ./figs/plot.eps (zeta_s convergence)")
    print("- ./figs/dnl.eps (DNL boxplot)")
    