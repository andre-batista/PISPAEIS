import sys
sys.path.insert(1, '../../eispy2d/library/')
import benchmark as bmk
import result as rst
import numpy as np
import matplotlib.pyplot as plt

# Configuration constants
FIGURE_SIZE = (12, 8)
BOXPLOT_SIZE = (5, 6)
CI_PLOT_SIZE = (10, 6)
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
    plt.close(fig)  # Close figure to free memory
    
    return fig

def create_dnl_boxplot(benchmark):
    """Create and style the DNL boxplot."""
    # Extract DNL data efficiently using list comprehension
    dnl_data = np.array([test.dnl for test in benchmark.testset.test[:30]])
    
    # Create boxplot with consistent figure size
    fig, ax = plt.subplots(figsize=BOXPLOT_SIZE)
    box_plot = ax.boxplot(dnl_data, vert=True, patch_artist=True)
    
    # Style boxplot elements
    box_plot['boxes'][0].set_facecolor('lightgray')
    box_plot['medians'][0].set_color('black')
    
    # Apply consistent font styling
    ax.set_ylabel('DNL', fontfamily=FONT_FAMILY, fontsize=FONT_SIZES['boxplot'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['boxplot'])
    ax.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels
    ax.grid(True, alpha=0.3)
    
    # Apply font family to tick labels
    for label in ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
    
    # Save with consistent configuration
    fig.tight_layout()
    fig.savefig('./figs/dnl.eps', **OUTPUT_CONFIG)
    plt.close(fig)
    
    return fig

def create_confidence_interval_plot(benchmark):
    """Create and style the confidence intervals plot."""
    fig, _ = benchmark.confint("zeta_s", show=False, paired=True, 
                               method=["osm","bim", "csi", "som"],
                               print_info=False)

    # Increase font size to 25 and remove title
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.set_xlabel(ax.get_xlabel(), fontsize=25)
        ax.set_ylabel(ax.get_ylabel(), fontsize=25)
        ax.set_title('')  # Remove title
        
        # Set y-axis labels to uppercase
        y_labels = [label.get_text().upper() for label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels)

    # Save and cleanup
    fig.tight_layout()
    fig.savefig('./figs/confidence_intervals.eps', **OUTPUT_CONFIG)
    plt.close(fig)
    
    return fig

# Main execution
if __name__ == "__main__":
    # Setup global configuration
    setup_matplotlib_defaults()
    
    try:
        # Load benchmark data
        benchmark = bmk.Benchmark(import_filename="average.bmk",
                                 import_filepath="../../data/shape/average/")
        
        # Create all plots
        plots_created = []
        
        # Generate plots with error handling
        try:
            zeta_fig = create_zeta_plot(benchmark)
            plots_created.append("./figs/plot.eps (zeta_s convergence)")
        except Exception as e:
            print(f"Error creating zeta plot: {e}")
        
        try:
            dnl_fig = create_dnl_boxplot(benchmark)
            plots_created.append("./figs/dnl.eps (DNL boxplot)")
        except Exception as e:
            print(f"Error creating DNL boxplot: {e}")
        
        try:
            ci_fig = create_confidence_interval_plot(benchmark)
            plots_created.append("./figs/confidence_intervals.eps (CI plot)")
        except Exception as e:
            print(f"Error creating confidence interval plot: {e}")
        
        # Report results
        if plots_created:
            print("Plots generated successfully:")
            for plot in plots_created:
                print(f"- {plot}")
        else:
            print("No plots were generated successfully.")
            
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
    
    finally:
        # Cleanup matplotlib to free memory
        plt.close('all')