# Performance Indicators for Shape and Position Assessment in Electromagnetic Inverse Scattering

<!-- [![IEEE Transactions on Antennas and Propagation](https://img.shields.io/badge/IEEE-TAP-blue.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

## Overview

This repository contains the complete research materials for the paper **"Performance Indicators for Shape and Position Assessment in Electromagnetic Inverse Scattering"** submitted to **IEEE Transactions on Antennas and Propagation**.

The research introduces two novel performance indicators specifically designed to evaluate electromagnetic inverse scattering algorithms:
- **Shape Error Indicator**: Quantifies geometric reconstruction accuracy
- **Position Error Indicator**: Measures object localization precision

## Abstract

Microwave imaging is a low-cost, non-invasive technique for detecting and characterizing objects in inaccessible media through electromagnetic field measurements. The evaluation of algorithms that solve the related electromagnetic inverse scattering problem typically relies on metrics such as Mean Square Error (MSE) or Structural Similarity (SSIM), which primarily assess contrast estimation accuracy but provide limited insight into the recovery of object geometry and position.

This paper introduces two novel performance indicators specifically designed to evaluate electromagnetic inverse scattering algorithms. The proposed indicators are applicable to both qualitative and quantitative methods, even when they are combined in an experiment. Comprehensive experimental validation is conducted using traditional algorithms: Linear Sampling Method (LSM), Orthogonality Sampling Method (OSM), Born Iterative Method (BIM), Contrast Source Iterative Method (CSI), Subspace Optimization Method (SOM), and Circle Approximation (CA).

## Repository Structure

```
PISPAEIS/
├── paper/                          # LaTeX source files for the manuscript
│   ├── main.tex                    # Main paper document
│   ├── mybib.bib                   # Bibliography file with 47 references
│   ├── main.pdf                    # Compiled PDF manuscript
│   ├── algorithms/                 # Algorithm descriptions
│   └── figs/                       # Paper figures and images
├── cover_letter/                   # Cover letter for journal submission
├── experiments/                    # Experimental implementations and results
│   ├── breast/                     # Breast phantom experiments
│   ├── shape/                      # Shape recovery studies
│   └── position/                   # Position detection studies
├── eispy2d/                        # Core algorithm implementations
│   └── library/                    # EISPY2D electromagnetic inverse scattering library
├── data/                           # Experimental datasets
│   ├── breast/                     # Breast phantom data
│   ├── shape/                      # Shape experiment data
│   └── position/                   # Position experiment data
├── requirements.txt                # Python dependencies
├── loaddata.py                     # Data loading utilities
└── README.md                       # This file
```

## Key Contributions

1. **Novel Performance Indicators**: Introduction of shape and position error indicators for electromagnetic inverse scattering evaluation
2. **Comprehensive Algorithm Comparison**: Evaluation of six traditional algorithms (LSM, OSM, BIM, CSI, SOM, CA)
3. **Multiple Experimental Frameworks**: Three distinct experimental setups demonstrating practical applicability
4. **Statistical Validation**: Robust statistical analysis supporting the experimental findings
5. **Open-Source Implementation**: Complete code and data for reproducibility

## Algorithms Evaluated

- **LSM**: Linear Sampling Method
- **OSM**: Orthogonality Sampling Method  
- **BIM**: Born Iterative Method
- **CSI**: Contrast Source Iterative Method
- **SOM**: Subspace Optimization Method
- **CA**: Circle Approximation

## Key Findings

The experiments revealed that:
- **Shape Recovery**: In scenarios with a single scatterer and low Degree of Nonlinearity, BIM achieves the best average performance in object geometry recovery
- **Position Detection**: In scenarios with a single small-sized, high-contrast scatterer, SOM and OSM achieve the best average performance in target localization

## Installation and Setup

### Prerequisites

- Python 3.12+
- LaTeX distribution (for paper compilation)
- Required Python packages (see `requirements.txt`)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/andre-batista/PISPAEIS.git
   cd PISPAEIS
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

1. **Load experimental data:**
   ```bash
   python loaddata.py
   ```

2. **Run specific experiments:**
   ```bash
   # Shape recovery experiments
   cd experiments/shape/star
   python runexperiment.py
   
   # Position detection experiments
   cd experiments/position/single
   python runexperiment.py
   
   # Breast phantom experiments
   cd experiments/breast
   python runexperiment.py
   ```

## Paper Compilation

The paper is written in LaTeX using the IEEE Transactions format.

### Compiling the PDF

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Data Description

### Experimental Datasets

- **Shape Recovery Data**: Various geometric shapes with different complexity levels
- **Position Detection Data**: Objects at different spatial locations
- **Breast Phantom Data**: Realistic medical imaging scenarios

### Data Format

All experimental data is stored in standardized formats compatible with the EISPY2D library.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{batista2025performance,
    title={Performance Indicators for Shape and Position Assessment in Electromagnetic Inverse Scattering},
    author={Batista, André Costa and Adriano, Ricardo and Batista, Lucas S.},
    year={2025},
    note={Preprint}
}
```

## Authors

- **André Costa Batista** - Department of Electrical Engineering, Universidade Federal de Minas Gerais
- **Ricardo Adriano** - Department of Electrical Engineering, Universidade Federal de Minas Gerais  
- **Lucas S. Batista** - Department of Electrical Engineering, Universidade Federal de Minas Gerais

## Acknowledgments

This work was supported in part by:
- Brazilian agency CAPES (Coordination for the Improvement of Higher Education Personnel) under Grant 88887.463864/2019-00
- FAPEMIG-CNPQ scholarship (process APQ-06716-24)
- CNPq (The National Council for Scientific and Technological Development)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this research, please contact:
- André Costa Batista: andre-costa@ufmg.br

---

**Keywords**: Algorithm evaluation, electromagnetic inverse scattering, microwave imaging, performance indicators