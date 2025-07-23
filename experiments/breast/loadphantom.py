import numpy as np
import pandas as pd
import os
from scipy.constants import epsilon_0

class BreastPhantomReader:
    """
    Reads and processes UW-Madison numerical breast phantom data
    based on the provided instruction manual.

    This class reads the breastInfo.txt, mtype.txt and pval.txt files,
    and calculates dielectric properties (dielectric constant and
    effective conductivity) at a specified frequency.
    """

    def __init__(self, directory_path):
        """
        Initializes the reader and loads phantom data.

        Args:
            directory_path (str): The path to the directory containing the
                                 phantom data files.
        """
        self.directory_path = directory_path
        self._verify_files()

        # Load metadata and grid data
        self._read_info()
        self._read_grid_data()

        # Store parameters and tissue mappings
        self._load_parameters()
        self._map_tissue_bounds()

        print(f"Phantom ID {self.breast_id} loaded successfully.")
        print(f"Dimensions (s1, s2, s3): ({self.s1}, {self.s2}, {self.s3})")
        print(f"Classification: {self.classification}")

    def _verify_files(self):
        """Verifies that all necessary files exist in the directory."""
        required_files = ['breastInfo.txt', 'mtype.txt', 'pval.txt']
        for filename in required_files:
            if not os.path.exists(os.path.join(self.directory_path, filename)):
                raise FileNotFoundError(
                    f"File '{filename}' not found in directory: {self.directory_path}"
                )

    def _read_info(self):
        """Reads the breastInfo.txt file to get dimensions and classification."""
        file_path = os.path.join(self.directory_path, 'breastInfo.txt')
        info = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                info[key.strip()] = value.strip()

        self.breast_id = info.get('breast ID')
        self.s1 = int(info.get('s1'))
        self.s2 = int(info.get('s2'))
        self.s3 = int(info.get('s3'))
        self.classification = int(info.get('classification'))

    def _read_grid_data(self):
        """
        Reads mtype.txt and pval.txt and reshapes them into 3D arrays.
        The 'F' order (Fortran-style) is used for reshaping, as specified
        by the file generation loop order in the manual. [cite: 25, 26, 27, 28, 30]
        """
        # Load mtype.txt
        mtype_path = os.path.join(self.directory_path, 'mtype.txt')
        mtype_flat = np.loadtxt(mtype_path)
        self.mtype_grid = mtype_flat.reshape((self.s1, self.s2, self.s3), order='F')

        # Load pval.txt
        pval_path = os.path.join(self.directory_path, 'pval.txt')
        pval_flat = np.loadtxt(pval_path)
        self.pval_grid = pval_flat.reshape((self.s1, self.s2, self.s3), order='F')

    def _load_parameters(self):
        """
        Loads Cole-Cole (Table 2) and Debye (Table 3) parameters
        into pandas DataFrames. [cite: 154, 279]
        """
        # Table 2 data: Single-pole Cole-Cole parameters [cite: 154]
        self.cole_cole_params = pd.DataFrame({
            'curve': ['minimum', 'group3-low', 'group3-median', 'group3-high', 'group1-low', 'group1-median', 'group1-high', 'maximum'],
            'eps_inf': [2.293, 2.908, 3.140, 4.031, 9.941, 7.821, 6.151, 1.000],
            'delta_eps': [0.141, 1.200, 1.708, 3.654, 26.60, 41.48, 48.26, 66.31],
            'tau_ps': [16.40, 16.88, 14.65, 14.12, 10.90, 10.66, 10.26, 7.585],
            'alpha': [0.251, 0.069, 0.061, 0.055, 0.003, 0.047, 0.049, 0.063],
            'sigma_s': [0.002, 0.020, 0.036, 0.083, 0.462, 0.713, 0.809, 1.370]
        }).set_index('curve')

        # Table 3 data: Single-pole Debye parameters (3-10 GHz band) [cite: 279]
        self.debye_params = pd.DataFrame({
            'curve': ['minimum', 'group3-low', 'group3-median', 'group3-high', 'group1-low', 'group1-median', 'group1-high', 'maximum', 'skin', 'muscle'],
            'eps_inf': [2.309, 2.848, 3.116, 3.987, 12.99, 13.81, 14.20, 23.20, 15.93, 21.66],
            'delta_eps': [0.092, 1.104, 1.592, 3.545, 24.40, 35.55, 40.49, 46.05, 23.83, 33.24],
            'tau_ps': [13.00, 13.00, 13.00, 13.00, 13.00, 13.00, 13.00, 13.00, 13.00, 13.00],
            'sigma_s': [0.005, 0.005, 0.050, 0.080, 0.397, 0.738, 0.824, 1.306, 0.831, 0.886]
        }).set_index('curve')
        # Alpha is 0 for single-pole Debye model (implicitly 1 in formula (1-alpha))
        self.debye_params['alpha'] = 0.0


    def _map_tissue_bounds(self):
        """
        Maps mtype media numbers to their upper and lower bound curve names.
        This mapping is inferred from Figures 2 and 4 and the manual text.
        """
        self.bounds_map = {
            # mtype: (lower_bound, upper_bound)
            1.1: ('group1-high', 'maximum'),
            1.2: ('group1-median', 'group1-high'),
            1.3: ('group1-low', 'group1-median'),
            2.0: ('group3-high', 'group1-low'),
            3.1: ('group3-median', 'group3-high'),
            3.2: ('group3-low', 'group3-median'),
            3.3: ('minimum', 'group3-low'),
        }

    def _calculate_complex_permittivity(self, freq_ghz, params):
        """Calculates complex permittivity using Cole-Cole formula."""
        omega = 2 * np.pi * freq_ghz * 1e9
        tau = params['tau_ps'] * 1e-12
        
        dispersive_term = params['delta_eps'] / (1 + (1j * omega * tau)**(1 - params['alpha']))
        conductive_term = params['sigma_s'] / (1j * omega * epsilon_0)
        
        # Complex relative permittivity is ε* = ε' - jε''
        # Where ε' = ε_inf + Re(dispersive_term) and ε'' = -Im(dispersive_term) - Im(conductive_term)
        # The manual groups static conductivity with the imaginary part.
        return params['eps_inf'] + dispersive_term + conductive_term

    def calculate_dielectric_properties(self, freq_ghz, model='cole-cole'):
        """
        Calculates dielectric constant and effective conductivity for each
        voxel in the grid at a specific frequency.

        Args:
            freq_ghz (float): The frequency of interest in GHz.
            model (str): The model to use, 'cole-cole' or 'debye'.

        Returns:
            tuple: A tuple containing two 3D NumPy arrays:
                   (dielectric_constant, effective_conductivity)
        """
        if model == 'cole-cole':
            params_table = self.cole_cole_params
        elif model == 'debye':
            params_table = self.debye_params
        else:
            raise ValueError("Model must be 'cole-cole' or 'debye'")

        omega = 2 * np.pi * freq_ghz * 1e9
        dims = (self.s1, self.s2, self.s3)
        complex_permittivity_grid = np.zeros(dims, dtype=complex)

        # Process normal breast tissues using interpolation
        for mtype, (lower_curve, upper_curve) in self.bounds_map.items():
            mask = self.mtype_grid == mtype
            if not np.any(mask):
                continue
            
            p_values = self.pval_grid[mask]
            lower_params = params_table.loc[lower_curve]
            upper_params = params_table.loc[upper_curve]

            # Interpolation for Debye model is applied to parameters
            if model == 'debye':
                interp_params = p_values[:, np.newaxis] * upper_params.values + \
                                (1 - p_values[:, np.newaxis]) * lower_params.values
                df_interp_params = pd.DataFrame(interp_params, columns=upper_params.index)
                eps_complex = self._calculate_complex_permittivity(freq_ghz, df_interp_params)
                complex_permittivity_grid[mask] = eps_complex
            else: # Interpolation for Cole-Cole is applied to final results
                eps_complex_lower = self._calculate_complex_permittivity(freq_ghz, lower_params)
                eps_complex_upper = self._calculate_complex_permittivity(freq_ghz, upper_params)
                eps_complex = p_values * eps_complex_upper + (1 - p_values) * eps_complex_lower
                complex_permittivity_grid[mask] = eps_complex

        # Process skin and muscle tissues (only for Debye model, as per Table 3)
        if model == 'debye':
            # Skin (mtype = -2) [cite: 44, 45]
            skin_mask = self.mtype_grid == -2
            if np.any(skin_mask):
                skin_params = params_table.loc['skin']
                complex_permittivity_grid[skin_mask] = self._calculate_complex_permittivity(freq_ghz, skin_params)
            
            # Muscle (mtype = -4) [cite: 46, 47]
            muscle_mask = self.mtype_grid == -4
            if np.any(muscle_mask):
                muscle_params = params_table.loc['muscle']
                complex_permittivity_grid[muscle_mask] = self._calculate_complex_permittivity(freq_ghz, muscle_params)

        # Extract dielectric constant and effective conductivity
        # ε* = ε' - jε''
        # Dielectric Constant (ε_r) = ε'
        # Effective Conductivity (σ_eff) = ωε₀ε'' = -ωε₀ * Im(ε*)
        dielectric_constant = np.real(complex_permittivity_grid)
        effective_conductivity = -np.imag(complex_permittivity_grid) * omega * epsilon_0

        return dielectric_constant, effective_conductivity


if __name__ == '__main__':
    # --- Usage Example ---
    # To run this example, create a directory called 'example_phantom_data'
    # and place the three data files (with sample content) inside it.

    # 1. Create directory and example data files
    example_dir = 'example_phantom_data'
    os.makedirs(example_dir, exist_ok=True)

    # breastInfo.txt content (based on manual example) [cite: 16]
    with open(os.path.join(example_dir, 'breastInfo.txt'), 'w') as f:
        f.write("breast ID=012204\n")
        f.write("s1=10\n") # Using small dimensions for example
        f.write("s2=10\n")
        f.write("s3=5\n")
        f.write("classification=2\n")

    # mtype.txt and pval.txt content (example data)
    # Total voxels = 10 * 10 * 5 = 500
    example_mtype = np.full(500, -1.0) # Immersion medium by default
    example_pval = np.zeros(500)
    
    # Define a small tissue region type 1.1 in the center
    example_grid = np.full((10, 10, 5), -1.0)
    example_pval_grid = np.zeros((10, 10, 5))
    example_grid[4:6, 4:6, 2] = 1.1 
    example_pval_grid[4:6, 4:6, 2] = 0.06468 # p-value from manual example

    # Save files in correct order (column-major)
    np.savetxt(os.path.join(example_dir, 'mtype.txt'), example_grid.flatten(order='F'), fmt='%1.1f')
    np.savetxt(os.path.join(example_dir, 'pval.txt'), example_pval_grid.flatten(order='F'), fmt='%.5f')

    try:
        # 2. Instantiate class with path to data
        reader = BreastPhantomReader(example_dir)

        # 3. Calculate properties at specific frequency (e.g., 6 GHz) [cite: 189]
        freq = 6.0  # GHz
        dielectric_const, effective_cond = reader.calculate_dielectric_properties(freq, model='cole-cole')

        # 4. Display results for a voxel of interest
        print(f"\n--- Properties Calculated at {freq} GHz ---")
        
        # Voxel (ii=4, jj=4, kk=2) that we defined as tissue 1.1
        # Remember Python indexing is 0-based
        ii, jj, kk = 4, 4, 2
        
        dc_voxel = dielectric_const[ii, jj, kk]
        ec_voxel = effective_cond[ii, jj, kk]
        
        print(f"Properties for example voxel (mtype={reader.mtype_grid[ii, jj, kk]}, pval={reader.pval_grid[ii, jj, kk]:.5f}):")
        print(f"  Dielectric Constant: {dc_voxel:.4f}")
        print(f"  Effective Conductivity: {ec_voxel:.4f} S/m")

        # Check an immersion voxel
        dc_immersion = dielectric_const[0, 0, 0]
        ec_immersion = effective_cond[0, 0, 0]
        print(f"\nProperties for an immersion voxel (mtype={reader.mtype_grid[0,0,0]}):")
        print(f"  Dielectric Constant: {dc_immersion:.4f} (expected ~0 or 1)")
        print(f"  Effective Conductivity: {ec_immersion:.4f} S/m (expected 0)")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")