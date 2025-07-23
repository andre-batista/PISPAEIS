import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

def loadphantom2D(filepath, f, epsilon_rb, sigma_b, new_res=None):
    """
    Load a 2D breast phantom with electromagnetic properties.
    
    Parameters:
    -----------
    filepath : str
        Path to the directory containing phantom data files
    f : float
        Frequency in Hz
    epsilon_rb : float
        Relative permittivity of background medium
    sigma_b : float
        Conductivity of background medium
    new_res : tuple or None
        New resolution (nx, ny) or (dx, dy) for resampling
    
    Returns:
    --------
    epsilon_r : ndarray
        Relative permittivity distribution
    sigma_eff : ndarray
        Effective conductivity distribution
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    mi : ndarray
        Indices of background medium pixels
    """
    
    # Constants
    omega = 2 * np.pi * f
    dx = 0.5e-3
    dy = 0.5e-3
    dz = 0.5e-3
    epsilon_0 = 8.854187817e-12
    
    # Cole-Cole parameters - 500MHz < f < 20GHz
    epsilon_inf = np.array([2.293, 2.908, 3.140, 4.031, 9.941, 7.821, 6.151, 1.])
    epsilon_delta = np.array([0.141, 1.2, 1.708, 3.654, 26.6, 41.48, 48.26, 66.31])
    tau = 1e-12 * np.array([16.4, 16.88, 14.65, 14.12, 10.9, 10.66, 10.26, 7.585])
    alpha = np.array([0.251, 0.069, 0.061, 0.055, 0.003, 0.047, 0.049, 0.063])
    sigma_s = np.array([0.002, 0.02, 0.036, 0.083, 0.462, 0.713, 0.809, 1.37])
    
    # Read breast information
    with open(os.path.join(filepath, 'breastInfo.txt'), 'r') as bi_file:
        breastInfo = bi_file.read().strip()
    
    # Read material type
    with open(os.path.join(filepath, 'mtype.txt'), 'r') as m_file:
        mi = np.fromfile(m_file, sep=' ')
    
    # Read property values
    with open(os.path.join(filepath, 'pval.txt'), 'r') as p_file:
        p_i = np.fromfile(p_file, sep=' ')
    
    # Calculate complex permittivity using Cole-Cole model
    epsilon_star = (epsilon_inf + 
                   epsilon_delta / (1 + (1j * omega * tau) ** (1 - alpha)) + 
                   sigma_s / (1j * omega * epsilon_0))
    
    bound_epsilon_r = np.real(epsilon_star)
    bound_sigma_eff = -np.imag(epsilon_star) * omega * epsilon_0
    
    # Initialize arrays
    epsilon_r = epsilon_rb * np.ones_like(mi)
    sigma_eff = sigma_b * np.ones_like(mi)
    
    # Tissue type mapping
    tissues = np.array([3.3, 3.2, 3.1, 2, 1.3, 1.2, 1.1])
    
    # Assign properties to different tissues
    for i in range(len(tissues)):
        mask = (mi == tissues[i])
        epsilon_r[mask] = (p_i[mask] * bound_epsilon_r[i+1] + 
                          (1 - p_i[mask]) * bound_epsilon_r[i])
        sigma_eff[mask] = (p_i[mask] * bound_sigma_eff[i+1] + 
                          (1 - p_i[mask]) * bound_sigma_eff[i])
    
    # Special tissue types
    epsilon_r[mi == -4] = np.max(bound_epsilon_r)
    sigma_eff[mi == -4] = np.max(bound_sigma_eff)
    
    epsilon_r[mi == -2] = np.mean(bound_epsilon_r)
    sigma_eff[mi == -2] = np.mean(bound_sigma_eff)
    
    # Parse dimensions from breastInfo
    i_pos = breastInfo.find('s1=')
    j_pos = breastInfo.find('s2=')
    k_pos = breastInfo.find('s3=')
    l_pos = breastInfo.find('class')
    
    I = int(breastInfo[i_pos+3:j_pos])
    J = int(breastInfo[j_pos+3:k_pos])
    K = int(breastInfo[k_pos+3:l_pos])
    
    # Reshape arrays
    epsilon_r = epsilon_r.reshape((I, J, K))
    sigma_eff = sigma_eff.reshape((I, J, K))
    
    # Extract 2D slice from middle
    epsilon_r = epsilon_r[I//2, :, :]
    sigma_eff = sigma_eff[I//2, :, :]
    
    # Find background medium indices
    mi = np.where((epsilon_r.flatten() == epsilon_rb) & 
                  (sigma_eff.flatten() == sigma_b))[0]
    
    # Update grid parameters
    dx = dy
    dy = dz
    I = J
    J = K
    
    Lx = I * dx
    Ly = J * dy
    
    # Resample if new resolution is specified
    if new_res is not None:
        if isinstance(new_res[0], int) and isinstance(new_res[1], int):
            # New resolution specified as number of points
            newi = new_res[0]
            newj = new_res[1]
            newdx = Lx / newi
            newdy = Ly / newj
        else:
            # New resolution specified as grid spacing
            newdx = new_res[0]
            newdy = new_res[1]
            newi = int(Lx / newdx)
            newj = int(Ly / newdy)
        
        # Create coordinate grids
        y = np.arange(J) * dy
        x = np.arange(I) * dx
        newy = np.arange(newj) * newdy
        newx = np.arange(newi) * newdx
        
        # Create meshgrid for new coordinates
        newY, newX = np.meshgrid(newy, newx, indexing='ij')
        new_points = np.column_stack([newX.ravel(), newY.ravel()])
        
        # Interpolate to new grid using RegularGridInterpolator
        f_epsilon = RegularGridInterpolator((x, y), epsilon_r, method='linear', bounds_error=False, fill_value=epsilon_rb)
        f_sigma = RegularGridInterpolator((x, y), sigma_eff, method='linear', bounds_error=False, fill_value=sigma_b)
        
        epsilon_r = f_epsilon(new_points).reshape(newi, newj)
        sigma_eff = f_sigma(new_points).reshape(newi, newj)
        
        # Update background indices
        mi = np.where((epsilon_r.flatten() == epsilon_rb) & 
                      (sigma_eff.flatten() == sigma_b))[0]
        
        dx = newdx
        dy = newdy
    
    # Handle NaN values
    epsilon_r[np.isnan(epsilon_r)] = epsilon_rb
    sigma_eff[np.isnan(sigma_eff)] = sigma_b
    
    return epsilon_r, sigma_eff, dx, dy, mi


# Example usage:
if __name__ == "__main__":
    # Example parameters
    filepath = "../../data/breast/"
    frequency = 1e9  # 1 GHz
    epsilon_rb = 1.0
    sigma_b = 0.0
    new_resolution = (100, 100)  # 100x100 grid
    
    # Load phantom
    epsilon_r, sigma_eff, dx, dy, mi = loadphantom2D(
        filepath, frequency, epsilon_rb, sigma_b, new_resolution
    )
    
    print(f"Phantom shape: {epsilon_r.shape}")
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
    print(f"Background pixels: {len(mi)}")
