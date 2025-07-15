from re import S
import error
import numpy as np
import evoalglib.representation as rpt
from abc import ABC, abstractmethod
from numpy import pi
from numpy import fft
from scipy.special import jv, yv
from scipy.special import hankel2 as h2v
from scipy.sparse import spdiags
from scipy.spatial.distance import pdist, squareform
from numba import jit
import sys
sys.path.insert(1, '..')
import configuration as cfg

CONTRAST_FIELD = 'contrast-field'
CONTRAST_SOURCE = 'contrast-source'
CSEB = 'cseb'
NIE = 'nie'
Y0 = 'y0'
Y0_NIE = 'yo-nie'

class ObjectiveFunction(ABC):
    def __init__(self):
        self.name = None
    def set_parameters(self, representation, scattered_field,
                       incident_field):
        self.representation = representation
        self.scattered_field = scattered_field
        self.incident_field = incident_field
    @abstractmethod
    def eval(self, x):
        pass
    @abstractmethod
    def __str__(self):
        return "Objective Function: "


class Rastrigin(ObjectiveFunction):
    def __init__(self, amplitude=10):
        super().__init__()
        self.name = 'rastringin'
        self.A = amplitude
        self.xopt = 0.
    def eval(self, x):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            raise error.WrongTypeInput('Rastrigin.eval', 'representation',
                                       'CanonicalProblems',
                                       str(type(self.representation)))
        super().eval(x)
        n = x.size
        x = self.representation.contrast(x)
        return self.A*n + np.sum(x**2-self.A*np.cos(2*pi*x))
    def __str__(self):
        message = super().__str__()
        message += 'Rastringin (Canonical, Nonlinear, Multimodal)\n'
        message += 'Amplitude: %.1f\n' % self.A
        message += 'Optimum solution: %.2f' % self.xopt
        return message


class Ackley(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.name = 'ackley'
        self.xopt = 0.
    def eval(self, x):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            raise error.WrongTypeInput('Ackley.eval', 'representation',
                                       'CanonicalProblems',
                                       str(type(self.representation)))
        super().eval(x)
        n = x.size
        x = self.representation.contrast(x)        
        return (-20*np.exp(-.2*np.sqrt(1/n*np.sum(x**2)))
                - np.exp(1/n*np.sum(np.cos(2*pi*x))) + 20 + np.exp(1))
    def __str__(self):
        message = super().__str__()
        message += 'Ackley (Canonical, Nonlinear, Multimodal)\n'
        message += 'Optimum solution: %.2f' % self.xopt
        return message


class Rosenbrock(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.name = 'rosenbrock'
        self.xopt = 0.
    def eval(self, x):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            raise error.WrongTypeInput('Rosenbrock.eval', 'representation',
                                       'CanonicalProblems',
                                       str(type(self.representation)))
        super().eval(x)
        n = x.size
        x = self.representation.contrast(x)     
        i = np.arange(n-1)
        return np.sum(100*(x[i]**2-x[i+1])**2 + (x[i]-1)**2)
    def __str__(self):
        message = super().__str__()
        message += 'Rosenbrock (Canonical, Nonlinear, Multimodal)\n'
        message += 'Optimum solution: %.2f' % self.xopt
        return message


class WeightedSum(ObjectiveFunction):
    def __init__(self, formulation=CONTRAST_FIELD, FFT=False, beta=6.,
                 regularizer=None):
        super().__init__()
        self.name = 'weighted_sum'
        self.FFT = FFT
        self.formulation = formulation
        self.beta = beta
        self.regularizer = regularizer
        
    def set_parameters(self, representation, scattered_field,
                       incident_field):
        self.representation = representation
        self.scattered_field = scattered_field
        self.incident_field = incident_field
        self.denominator_data = np.sum(np.abs(scattered_field)**2)
        self.denominator_space = np.sum(np.abs(incident_field)**2)
        if self.FFT:
            self.GE = get_extended_matrix(
                representation.discretization.configuration,
                representation.discretization.elements
            )
        else:
            self.GE = None
        if self.formulation == CSEB:
            self.G = get_cseb_matrix(
                representation.discretization.configuration,
                representation.discretization.elements
            )
        if self.formulation == Y0 or self.formulation == Y0_NIE:
            kb = representation.discretization.configuration.kb
            J = self.regularizer.solve(K=representation.discretization.GS,
                                       y=scattered_field)
            F, G = get_y0_matrices(representation.discretization.configuration,
                                   representation.discretization.elements, J)
            self.G = G
            self.Eh = incident_field - 1j*kb**2/4*F

    def eval(self, x):
        X = self.representation.contrast(x)
        E = self.representation.total_field(
            x, self.representation.discretization.elements
        )
        Es = self.scattered_field
        Ei = self.incident_field
        GS = self.representation.discretization.GS
        GD = self.representation.discretization.GD
        if self.formulation == CONTRAST_FIELD and not self.FFT:
            J = spdiags(X.flatten(), 0, X.size, X.size) @ E
            data_res = Es-GS@J
            state_res = E-Ei-GD@J
            data_res = self.representation.discretization.residual_data(
                self.scattered_field, contrast=X, total_field=E
            )
            state_res = self.representation.discretization.residual_state(
                self.incident_field, contrast=X, total_field=E
            )
        elif self.formulation == CONTRAST_FIELD and self.FFT:
            X = spdiags(X.flatten(), 0, X.size, X.size)
            J = X @ E
            data_res = Es-GS@J
            state_res = (E - Ei - fft_multiplication(
                self.GE, J, self.representation.discretization.elements
            ))
        elif self.formulation == CONTRAST_SOURCE and not self.FFT:
            X = spdiags(X.flatten(), 0, X.size, X.size) 
            J = X @ E
            data_res = Es-GS@J
            state_res = J-X@Ei-X@GD@J
        elif self.formulation == CONTRAST_SOURCE and self.FFT:
            X = spdiags(X.flatten(), 0, X.size, X.size) 
            J = X @ E
            data_res = Es-GS@J
            state_res = (J - X@Ei - fft_multiplication(
                self.GE, J, self.representation.discretization.elements
            ))
        elif self.formulation == CSEB and not self.FFT:
            X = X.flatten()
            J = spdiags(X, 0, X.size, X.size) @ E
            p = X/(1-X*self.G.diagonal(0))
            p = spdiags(p, 0, p.size, p.size)
            data_res = Es-GS@J
            state_res = J - p @ Ei - p @ (GD @ J  - self.G @ J)
        elif self.formulation == CSEB and self.FFT:
            X = X.flatten()
            J = spdiags(X, 0, X.size, X.size) @ E
            p = X/(1-X*self.G.diagonal(0))
            p = spdiags(p, 0, p.size, p.size)
            resolution = self.representation.discretization.elements
            data_res = Es-GS@J
            state_res = J - p @ Ei - p @ (fft_multiplication(
                self.GE, J, resolution
            ) @ J - self.G @ J)
        elif self.formulation == NIE and not self.FFT:
            X = X.flatten()
            J = spdiags(X, 0, X.size, X.size) @ E
            R = self.beta*X/(self.beta*X + 1)
            R = spdiags(R, 0, R.size, R.size)
            betaW = self.beta*J
            data_res = Es-GS@J
            state_res = betaW - R @ Ei - R @ (betaW + 1/self.beta * GD @ betaW)
        elif self.formulation == NIE and self.FFT:
            resolution = self.representation.discretization.elements
            X = X.flatten()
            J = spdiags(X, 0, X.size, X.size) @ E
            R = self.beta*X/(self.beta*X + 1)
            R = spdiags(R, 0, R.size, R.size)
            betaW = self.beta*J
            data_res = Es-GS@J
            state_res = (betaW - R @ Ei
                         - R @ (betaW + fft_multiplication(self.GE, J,
                                                           resolution)))
        elif self.formulation == Y0:
            X = spdiags(X.flatten(), 0, X.size, X.size)
            J = X @ E
            data_res = Es-GS@J
            state_res = J - X @ self.Eh - X @ self.G @ J
        elif self.formulation == Y0_NIE:
            X = X.flatten()
            J = spdiags(X, 0, X.size, X.size) @ E
            R = self.beta*X/(self.beta*X + 1)
            R = spdiags(R, 0, R.size, R.size)
            betaW = self.beta*J
            data_res = Es-GS@J
            state_res = betaW - R @ self.Eh - R @ (betaW + 1/self.beta
                                                   * self.G @ betaW)
        return (np.sum(np.abs(data_res)**2)/self.denominator_data
                + np.sum(np.abs(state_res)**2)/self.denominator_space)
    def __str__(self):
        message = super().__str__()
        message += 'Weighted Sum of Data and State Equations Residuals'
        message += '\nFormulation: ' + self.formulation
        return message


def get_extended_matrix(configuration, resolution):
    xmin, xmax = cfg.get_bounds(configuration.Lx)
    ymin, ymax = cfg.get_bounds(configuration.Ly)
    NX, NY = resolution[1], resolution[0]
    dx, dy = configuration.Lx/NX, configuration.Ly/NY
    [xe, ye] = np.meshgrid(np.arange(xmin-(NX/2-1)*dx, xmax+NY/2*dx, dx),
                           np.arange(ymin-(NY/2-1)*dy, ymax+NY/2*dy, dy))
    Rmn = np.sqrt(xe**2 + ye**2)
    kb = configuration.kb
    a = np.sqrt(dx*dy/pi)
    Gmn = -1j*pi*kb*a/2*jv(1, kb*a)*h2v(0, kb*Rmn)
    Gmn[NY-1, NX-1] = -(1j/2)*(pi*kb*a*h2v(1, kb*a)-2j)
    GE = np.zeros((2*NY-1, 2*NX-1), dtype=complex)
    GE[:NY, :NX] = Gmn[NY-1:2*NY-1, NX-1:2*NX-1]
    GE[NY:2*NY-1, NX:2*NX-1] = Gmn[:NY-1, :NX-1]
    GE[NY:2*NY-1, :NX] = Gmn[:NY-1, NX-1:2*NX-1]
    GE[:NY, NX:2*NX-1] = Gmn[NY-1:2*NY-1, :NX-1]
    return GE
    

def fft_multiplication(GE, J, resolution):
    NS = J.shape[1]
    Es = np.zeros(J.shape, dtype=complex)
    NY, NX = resolution
    for s in range(NS):
        Je = np.zeros((2*NY-1, 2*NX-1), dtype=complex)
        Je[:NY, :NX] = J[:, s].reshape(resolution)
        aux = fft.ifft2(fft.fft2(GE)*fft.fft2(Je))
        Es[:, s] = aux[:NY, :NX].flatten()
    return Es


def get_cseb_matrix(configuration, resolution):
    kb = configuration.kb
    x, y = cfg.get_coordinates_ddomain(
            configuration=configuration,
            resolution=resolution
    )
    a = np.sqrt((x[0, 1]-x[0, 0])*(y[1, 0]-y[0, 0])/np.pi)
    G = -1j*np.pi*kb*a/2*h2v(1, kb*a)*jv(0, kb*np.sqrt(x**2+y**2))
    G = spdiags(G.flatten(), 0, G.size, G.size)
    return G


def get_y0_matrices(configuration, resolution, J):
    kb = configuration.kb
    x, y = cfg.get_coordinates_ddomain(configuration=configuration,
                                       resolution=resolution)
    dx, dy = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0]
    a = np.sqrt(dx*dy/np.pi)
    R = squareform(pdist(np.transpose(np.vstack((x.flatten(), y.flatten())))))
    G = np.zeros(R.shape, dtype=complex)
    G[R!=0] = - np.pi*a*kb/2*jv(1, kb*a)*yv(0, kb*R[R!=0])
    np.fill_diagonal(G, -1/2*(np.pi*a*kb*yv(1, kb*a) + 2))
    J0 = jv(0, kb*R)
    F = np.zeros((x.size, configuration.NS), dtype=complex)
    _compute_F(F, J0, J, resolution, dy, dx)
    return F, G
    

@jit(nopython=True)
def _compute_F(F, J0, J, resolution, dy, dx):
    for s in range(F.shape[1]):
        for n in range(F.shape[0]):
            A = np.reshape(J0[n, :] * J[:, s], resolution)
            F[n, s] = np.trapz(np.trapz(A, dx=dy), dx=dx)