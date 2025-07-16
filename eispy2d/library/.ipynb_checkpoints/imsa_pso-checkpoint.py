r"""An Integrated Multiscaling Strategy Based on a Particle Swarm Algorithm.

This module implements a qualitative and stochastic method for solving
the 2D eletromagnetic inverse scattering (EIS) problem. The method
integrates a multiscale strategy with a global optimization method. The
multiscale strategy is a procedure in which the image is divided in
multiples Regions of Interest (RoI) by detecting and locating scatters.
This procedure is integrated with the Particle Swarm Optimization (PSO)
algorithm which is responsible for determining field and contrast at the
RoI's. This is a population-based method which may avoid local minima,
a common problem in nonlinear and nonconvex problems such as EIS.

This module provides the following class:

    :class:`IMSA_PSO`
        The Iterative Multiscaling Approach - Particle Swarm
        Optimization (IMSA-PSO) algorithm.

This module provides the following auxiliar functions:

    :func:`name`
        Description.

'References
----------
.. [1] Caorsi, Salvatore, et al. "A new methodology based on an
   iterative multiscaling for microwave imaging." IEEE transactions
   on microwave theory and techniques 51.4 (2003): 1162-1173.'
.. [2] Caorsi, Salvatore, et al. "Location and imaging of
   two-dimensional scatterers by using a particle swarm algorithm."
   Journal of Electromagnetic Waves and Applications 18.4 (2004):
   481-494.
.. [3] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa. "Detection,
   location, and imaging of multiple scatterers by means of the
   iterative multiscaling method." IEEE transactions on microwave theory
   and techniques 52.4 (2004): 1217-1228.
.. [4] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa. "Analysis
   of the stability and robustness of the iterative multiscaling
   approach for microwave imaging applications." Radio science 39.5
   (2004): 1-17.
.. [5] Donelli, Massimo, and Andrea Massa. "Computational approach based
   on a particle swarm optimizer for microwave imaging of
   two-dimensional dielectric scatterers." IEEE Transactions on
   Microwave Theory and Techniques 53.5 (2005): 1761-1776.
.. [6] Donelli, Massimo, et al. "An integrated multiscaling strategy
   based on a particle swarm algorithm for inverse scattering problems."
   IEEE Transactions on Geoscience and Remote Sensing 44.2 (2006):
   298-312.
.. [7] Donelli, Massimo, et al. "Three-dimensional microwave imaging
   problems solved through an efficient multiscaling particle swarm
   optimization." IEEE Transactions on Geoscience and remote sensing
   47.5 (2008): 1467-1481.
.. [8] Salucci, Marco, et al. "Multifrequency particle swarm
   optimization for enhanced multiresolution GPR microwave imaging."
   IEEE Transactions on Geoscience and Remote Sensing 55.3 (2016):
   1305-1317.
"""

# Standard libraries
import sys
import pickle
import multiprocessing
import time as tm
import copy as cp
import numpy as np
from numpy import pi
from numpy import random as rnd
from scipy.sparse import dia_matrix
from scipy.linalg import norm
from scipy.special import jv, jvp, hankel2, h2vp
from scipy.interpolate import interp2d
from scipy.constants import epsilon_0
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

# Developed libraries
import configuration as cfg
import inputdata as ipt
import results as rst
import solver as slv
import forward as fwr
import mom_cg_fft as mom
import experiment as exp
import error


class IMSA_PSO(slv.Solver):
    r"""The Iterative Multiscalling Particle Swarm Opimization algorithm.

    This class is the implementation of a Particle Swarm Optimization
    (PSO) embedded in an Iterative Multiscaling Approach (IMSA) for 2D
    Electromagnetic Inverse Scattering (EIS) problem.

    This is a qualitative and stochastic method which iteratively
    reduces the image in multiple regions of interest (RoI) and solves
    the intern total electric field and contrast map through a
    population-based optimization algorithm.

    The main reference for this implementation is [8]_ although the
    description of the clustering process is better described in [1]_
    and [2]_. Furthermore, the GPR version of EIS problem is considered
    in [8]_, which is not the case in this implementation. However,
    we are following the nomenclature define in [8]_. So whoever reads
    the paper will find the same name for the variables defined in this
    code. Also, we are not considering the multifrequency approach since
    the whole architecture of this project is not prepared for this
    case. Therefore, this implementation is similar to one presented in
    [5]_ and [6]_ while it follows the nomenclature presented in [8]_.

    Attributes
    ----------
        configuration : :class:`configuration.Configuration`
            Constants of the problem configuration.

        w, c1, c2 : float, default: 2.0, 2.0, and 0.4
            Inertial weight and acceleration coefficients for velocity
            update.

        number_particles : int, default: 50
            Size of the particles population.

        number_iterations : int or list of int, default: 20000
            Maximum number of iterations allowed in the PSO algorithm
            for each scatling process. If a single integer value is
            passed, then it is assumed that this number is the same for
            all scaling process.

        resolution : tuple of list of tuple
            Image size in pixels for each scaling step. If a single
            tuple is passed, then it is assumed that the it is constant.
            If `resolution` and `number_iterations` are lists, then
            *they must have the same length*.

        Phi_th : float, default: `-numpy.inf`
            Threshold value for the objective function which is used as
            stop criterion for the PSO algorithm.

        eta : float, default: `-numpy.inf`
            Threshold value for the stop criterion for the scaling
            process defined as:

            .. math:: \frac{A_s-A_{s-1}}{A_s} \leq \eta

        maximum_evaluations : int, default: `numpy.inf`
            A threshold value for the number of evaluations of particles
            throughout the whole algorithm. This is not cover in the
            original paper but it was added for further comparisons.

        alias : str, default: 'imsapso'
            A nickname for the object. This may be useful when comparing
            different configuration of parameters of this algorithm.

        forward : :class:`forward.Forward`
            Forward solver object used only for computing the incident
            field. Default: :class:`mom_cg_fft.MoM_CG_FFT`.

    Methods
    -------
        :func:`solve`
            Run an instance.

    References
    ----------
    .. [1] Caorsi, Salvatore, et al. "A new methodology based on an
       iterative multiscaling for microwave imaging." IEEE transactions
       on microwave theory and techniques 51.4 (2003): 1162-1173.
    .. [2] Caorsi, Salvatore, et al. "Location and imaging of
       two-dimensional scatterers by using a particle swarm algorithm."
       Journal of Electromagnetic Waves and Applications 18.4 (2004):
       481-494.
    .. [3] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa.
       "Detection, location, and imaging of multiple scatterers by means
       of the iterative multiscaling method." IEEE transactions on
       microwave theory and techniques 52.4 (2004): 1217-1228.
    .. [4] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa.
       "Analysis of the stability and robustness of the iterative
       multiscaling approach for microwave imaging applications." Radio
       science 39.5 (2004): 1-17.
    .. [5] Donelli, Massimo, and Andrea Massa. "Computational approach
       based on a particle swarm optimizer for microwave imaging of
       two-dimensional dielectric scatterers." IEEE Transactions on
       Microwave Theory and Techniques 53.5 (2005): 1761-1776.
    .. [6] Donelli, Massimo, et al. "An integrated multiscaling strategy
       based on a particle swarm algorithm for inverse scattering
       problems." IEEE Transactions on Geoscience and Remote Sensing
       44.2 (2006): 298-312.
    .. [7] Donelli, Massimo, et al. "Three-dimensional microwave imaging
       problems solved through an efficient multiscaling particle swarm
       optimization." IEEE Transactions on Geoscience and remote sensing
       47.5 (2008): 1467-1481.
    .. [8] Salucci, Marco, et al. "Multifrequency particle swarm
       optimization for enhanced multiresolution GPR microwave imaging."
       IEEE Transactions on Geoscience and Remote Sensing 55.3 (2016):
       1305-1317.
    """

    def __init__(self, configuration, resolution, c1=2., c2=2., w=.4,
                 number_particles=50, number_iterations=20000, alias='imsapso',
                 cost_function_threshold=-np.inf, area_threshold=-np.inf,
                 max_iterations_without_improvement=np.inf,
                 maximum_evaluations=np.inf, forward_solver=None):
        """Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                Configuration object.

            resolution : tuple or list of tuples
                Image size for each scaling process. If a single tuple
                is passed, then the number of scaling steps will be
                determined by the length of the `number_iterations`
                input and the resolution will be constant. If
                `resolution` and `number_iterations` are passed as
                lists, then *they must have the same length*.

            c1, c2, w : float, default: 2.0, 2.0, 0.4
                Acceleration coefficients and inertial weight for
                velocity update.

            number_particles : int, default: 50
                Size of the population.

            number_iterations : int, default: 20000
                Maximum number of iterations in PSO. If a single value
                is passed, the number of scaling steps will be
                determined by the length of the `resolution` variable.
                If `resolution` and `number_iterations` are passed as
                lists, then *they must have the same length*.

            alias : str, default: 'imsapso'
                A nickname for the object. It may be useful when
                comparing different versions of the algorithm.

            cost_function_threshold : float, default: -`numpy.inf`
                Threshold value for the objective function evaluation
                of the global best particle during the PSO algorithm.
                This is a stop criterion (page 4, right column, 2-c
                topic).

            area_threshold : float, default: -`numpy.inf`
                Stop criterion for the scaling process (page 5, left
                column, 3-b topic.).

            max_iterations_without_improvement : int, default: `numpy.inf`
                Stop criterion of PSO algorithm. If no improvement in
                the global particle is found in some number of
                interations, then PSO stops.

            maximum_evalutations : int, default: `numpy.inf`
                Maximum number of evaluations allowed in the whole
                algorithm. It is not cover in the original paper but it
                may be a useful tool when comparing against different
                algorithms.

            forward_solver : :class:`forward.Forward`
                Forward solver object used only for computing the
                incident field. Default: :class:`mom_cg_fft.MoM_CG_FFT`.
        """
        # Validating inputs
        if type(resolution) is not tuple and type(resolution) is not list:
            raise error.WrongTypeInput('IMSA_PSO.__init__', 'resolution',
                                       'tuple of list of tuple',
                                       type(resolution))
        if (type(number_iterations) is not int
                and type(number_iterations) is not list):
            raise error.WrongTypeInput('IMSA_PSO.__init__',
                                       'number_iterations',
                                       'int of list of int',
                                       type(number_iterations))
        if (type(resolution) is list and type(number_iterations) is list
                and len(resolution) != len(number_iterations)):
            raise error.WrongValueInput('IMSA_PSO.__init__',
                                        'number_iterations', 'same length'
                                        + ' than resolution', 'length %d'
                                        % len(number_iterations))

        # Base class builder
        super().__init__(configuration)

        # Set attributes
        self.c1, self.c2, self.w = c1, c2, w
        self.name = 'IMSA-PSO'
        self.number_particles = number_particles
        self.Phi_th = cost_function_threshold
        self.eta = area_threshold
        self.maximum_evaluations = maximum_evaluations
        self.alias = alias
        self.miwi = max_iterations_without_improvement

        # Make sure that the resolution attribute is a list
        if type(resolution) is tuple:
            self.resolution = [resolution]
        else:
            self.resolution = resolution.copy()

        # Make sure that the number_iterations attribute is a list
        if type(number_iterations) is int:
            self.number_iterations = [number_iterations] * len(self.resolution)
        elif type(number_iterations) is float:
            self.number_iterations = ([int(number_iterations)]
                                      * len(self.resolution))
        elif type(number_iterations) is list:
            if len(number_iterations) > 1:
                if len(self.resolution) == 1:
                    self.resolution = self.resolution * len(number_iterations)
                self.number_iterations = number_iterations.copy()
            else:
                self.number_iterations = number_iterations*len(self.resolution)

        # Set foward solver
        if forward_solver is None:
            self.forward = mom.MoM_CG_FFT(self.configuration)
        else:
            self.forward = forward_solver

    def solve(self, instance, print_info=True, print_file=sys.stdout,
              number_executions=30, max_contrast=100, min_contrast=0,
              run_parallelly=True, store_image='best', percent_step=10.,
              save_executions=None,
              file_path=''):
        """Run an instance.

        This is the method which must be called to run an instance.
        Since this is a stochastic algorithm, then this method provides
        the option to run the instance multiple times and meausure the
        performance by means of the mean among the executions. This is
        statiscaly more robust in order to analyse the algorithm and it
        is not covered in the original paper [1]_.

        Parameters
        ----------
            instance : :class:`inputdata.InputData`
                Object containing the instance data (scattered field,
                resolution of the final image etc).

            print_info : bool, default: True
                Print or not information.

            print_file, default: `sys.stdout`
                An object with a write method for printing information.

            number_executions : int, default: 30
                Number of times in which the instance will be run.

            min_contrast : complex or float, defaut: 0
                Minimum contrast allowed. This variable limits the
                search space of contrast variables (page 5, left column,
                footnote 3). For problems with perfect dielectric
                objects, this argument may be float. Otherwise, it must
                be complex.

            max_contrast : float, default: None
                Maximum contrast allowed. This variable limits the
                search space of contrast variables (page 5, left column,
                footnote 3). For problems with perfect dielectric
                objects, this argument may be float. Otherwise, it must
                be complex.

            run_parallelly : bool, default: True
                Run executions in parallel.

            store_image : {'best', 'worst', 'median'}, default: 'best'
                Define the criterion for choosing one of the executions
                for storing the contrast map and total field at the
                :class:`results.Results` object. So you may store the
                best case, the worst one or the median one.

            percent_step : float, default: 10.0
                The convergence of the objective function and others
                measures will be recorded according to the percentage
                of iterations, i.e., 0%, 10%, 20%, so on. This will make
                the comparison among different number of iterations more
                reasonable. This variable determines the step of the
                percentage.

            save_executions : str, default: None
                If you want to save the results of each execution for
                posteriori analysis, give a string for the name of the
                file in which the data will be saved. The data will be
                save as a list of :class:`results.Results`.

            file_path : str, default: ''
                Path to the directory in which the results of the
                executions will be saved.

        Returns
        -------
            :class:`results.Results`
                The object with the best/worst/median recovered map and
                the mean values of measures.

        References
        ----------
            .. [1] Salucci, Marco, et al. "Multifrequency particle swarm
               optimization for enhanced multiresolution GPR microwave
               imaging." IEEE Transactions on Geoscience and Remote
               Sensing 55.3 (2016): 1305-1317.
        """
        # Validate inputs
        if (store_image != 'best' and store_image != 'worst'
                and store_image != 'median'):
            raise error.WrongValueInput('IMSA_PSO.solve', 'store',
                                        'best/worst/median', store_image)

        # Base class builder
        super().solve(instance, print_info, print_file)

        # Initialize result object
        result = rst.Results(instance.name + '_' + self.alias,
                             method_name=self.alias,
                             configuration_filename=self.configuration.name,
                             configuration_filepath=self.configuration.path,
                             input_filename=instance.name,
                             input_filepath=instance.path)

        # Print information
        if print_info:
            text = self.__execution_info(number_executions, min_contrast,
                                         max_contrast, store_image,
                                         percent_step, save_executions)
            print(text, file=print_file)

        # Compute the incident field
        Psi_i = self.forward.incident_field(instance.resolution)

        # Initialize list for storing executions
        executions, fitness = [], []

        if run_parallelly:
            if print_info:
                print('Running executions in parallel...', file=print_file)

            num_cores = multiprocessing.cpu_count()
            output = (Parallel(n_jobs=num_cores))(delayed(self.run_algorithm)
                                                  (instance, Psi_i,
                                                   max_contrast, min_contrast,
                                                   False, None)
                                                  for ne in
                                                  range(number_executions))
        else:
            output = []
            for ne in range(number_executions):
                output.append(self.run_algorithm(instance, Psi_i,
                                                 max_contrast, min_contrast,
                                                 print_info, print_file))

        # Append results of executions
        for ne in range(number_executions):
            executions.append(output[ne][0])
            fitness.append(output[ne][1])

        # Find the execution which will be stored
        store_idx, store_fx = 0, fitness[0][-1]
        median = [fitness[0][-1]]
        execution_time = [executions[0].execution_time]
        for ne in range(1, number_executions):
            if store_image == 'best' and fitness[ne][-1] < store_fx:
                store_idx, store_fx = ne, fitness[ne][-1]
            elif store_image == 'worst' and fitness[ne][-1] > store_fx:
                store_idx, store_fx = ne, fitness[ne][-1]
            elif store_image == 'median':
                median.append(fitness[ne][-1])
            execution_time.append(executions[ne].execution_time)
        if store_image == 'median':
            med = np.median(median)
            for ne in range(number_executions):
                if median[ne] == med:
                    store_fx, store_idx = median[ne], ne
                    break

        # Record the total and scattered field, relative permittivity,
        # conductivity and average of execution_time
        result.et = executions[store_idx].et
        result.es = executions[store_idx].es
        result.epsilon_r = executions[store_idx].epsilon_r
        result.sigma = executions[store_idx].sigma
        result.execution_time = np.mean(execution_time)

        # Compute means of measures per percent of iterations
        iterations = np.arange(0, 100+percent_step, percent_step)
        result.objective_function = self.__get_means(fitness, iterations)
        if len(executions[0].zeta_rn) > 0:
            result.zeta_rn = self.__get_means(self.__get_values('zeta_rn',
                                                                executions),
                                              iterations)
        if len(executions[0].zeta_rpad) > 0:
            result.zeta_rpad = self.__get_means(self.__get_values('zeta_rpad',
                                                                  executions),
                                                iterations)
        if len(executions[0].zeta_epad) > 0:
            result.zeta_epad = self.__get_means(self.__get_values('zeta_epad',
                                                                  executions),
                                                iterations)
        if len(executions[0].zeta_ebe) > 0:
            result.zeta_ebe = self.__get_means(self.__get_values('zeta_ebe',
                                                                 executions),
                                               iterations)
        if len(executions[0].zeta_eoe) > 0:
            result.zeta_eoe = self.__get_means(self.__get_values('zeta_eoe',
                                                                 executions),
                                               iterations)
        if len(executions[0].zeta_sad) > 0:
            result.zeta_sad = self.__get_means(self.__get_values('zeta_sad',
                                                                 executions),
                                               iterations)
        if len(executions[0].zeta_sbe) > 0:
            result.zeta_sbe = self.__get_means(self.__get_values('zeta_sbe',
                                                                 executions),
                                               iterations)
        if len(executions[0].zeta_soe) > 0:
            result.zeta_soe = self.__get_means(self.__get_values('zeta_soe',
                                                                 executions),
                                               iterations)
        if len(executions[0].zeta_be) > 0:
            result.zeta_be = self.__get_means(self.__get_values('zeta_be',
                                                                executions),
                                              iterations)
        if len(executions[0].zeta_tfmpad) > 0:
            result.zeta_tfmpad = self.__get_means(
                self.__get_values('zeta_tfmpad', executions), iterations
            )
        if len(executions[0].zeta_tfppad) > 0:
            result.zeta_tfppad = self.__get_means(
                self.__get_values('zeta_tfppad', executions), iterations
            )

        # Save results of all executions
        if save_executions is not None:
            with open(file_path + save_executions, 'wb') as datafile:
                pickle.dump(executions, datafile)

        return result

    def run_algorithm(self, instance, Psi_i, tau_max=None, tau_min=None,
                      print_info=True, print_file=sys.stdout):
        """Run single execution of IMSA-PSO.

        This method is not intended to be called by the user. It is only
        an auxiliar method for parallel running.

        The name of the variables are according to [1]_.

        Parameters
        ----------
            instance : :class:`inputdata.InputData`
                Object containing the instance data (scattered field,
                resolution of the final image etc).

            Psi_i : :class:`numpy.ndarray`
                Incident field matrix.

            tau_min : complex or float, defaut: None
                Minimum contrast allowed. This variable limits the
                search space of contrast variables (page 5, left column,
                footnote 3). For problems with perfect dielectric
                objects, this argument may be float. Otherwise, it must
                be complex.

            tau_max : float, default: None
                Maximum contrast allowed. This variable limits the
                search space of contrast variables (page 5, left column,
                footnote 3). For problems with perfect dielectric
                objects, this argument may be float. Otherwise, it must
                be complex.

            print_info : bool, default: True
                Print or not information.

            print_file, default: `sys.stdout`
                An object with a write method for printing information.

        Returns
        -------
            :class:`results.Results`

        References
        ----------
            .. [1] Salucci, Marco, et al. "Multifrequency particle swarm
               optimization for enhanced multiresolution GPR microwave
               imaging." IEEE Transactions on Geoscience and Remote
               Sensing 55.3 (2016): 1305-1317.
        """
        # Validate inputs
        if instance.resolution[0]*instance.resolution[1] != Psi_i.shape[0]:
            raise error.WrongValueInput('IMSA_PSO.run_algorithm', 'Psi_i',
                                        'same number of rows than the '
                                        + 'number of elements in the final '
                                        + 'resolution', 'Psi_i.shape = '
                                        + str(Psi_i.shape) + ', final '
                                        + 'resolution = '
                                        + str(instance.resolution))

        # Define variables
        result = rst.Results(name='',
                             configuration_filename=self.configuration.name,
                             configuration_filepath=self.configuration.path)
        Psi_s = np.copy(instance.es)  # Scattered field
        final_resolution = instance.resolution  # Recovered image resolution
        c1, c2, w = self.c1, self.c2, self.w  # Velocity parameters
        eta, Phi_th = self.eta, self.Phi_th  # Stop criteria
        I = self.number_iterations.copy()
        P = self.number_particles
        V = self.configuration.NS  # Number of incidences
        MAX_EVALS = self.maximum_evaluations
        MIWI = self.miwi
        resolution = self.resolution.copy()  # Resolution for each scale step
        S = len(resolution)  # Number of scaling steps

        # Determine the type of the contrast particles
        if (self.configuration.perfect_dielectric
                or self.configuration.good_conductor):
            contrast_type = float
        else:
            contrast_type = complex

        # Mesh of the indicent field data
        xi, yi = cfg.get_coordinates_ddomain(resolution=final_resolution,
                                             configuration=self.configuration)

        # Mesh variables for the first iteration
        xmin, xmax = cfg.get_bounds(self.configuration.Lx)
        ymin, ymax = cfg.get_bounds(self.configuration.Ly)
        xm, ym = cfg.get_coordinates_sdomain(self.configuration.Ro,
                                             self.configuration.NM)
        kb = self.configuration.kb

        # Left-bottom and right-top points that detemines the RoI for
        # the first iteration
        p0, p1 = [(xmin, ymin)], [(xmax, ymax)]
        last_x, last_y, last_region_ref = [], [], []

        # The RoI corresponding area
        A = [self.configuration.Lx*self.configuration.Ly]

        # Variables that will be used during the iterations
        convergence = []  # Cost function evaluation array
        execution_time = 0.
        number_evaluations = 0
        max_evals_stop = False

        if print_info:
            print('----------------------------------------', file=print_file)

        # Start IMSA-PSO
        for s in range(S):

            # Compute Green function array for each RoI
            tic = tm.time()
            N, R = resolution[s][0]*resolution[s][1], len(p0)
            x, y, cell_area = compute_mesh(p0, p1, resolution[s])
            Gext, Gint = green_function(xm, ym, x, y, cell_area, kb,
                                        resolution[s])

            # The incident field must be estimated in each region. Here,
            # we are interpolating
            Psi_i_r = []
            for r in range(R):
                Psi_i_r.append(np.zeros((N, V), dtype=complex))
                for v in range(V):
                    Psi_i_r[-1][:, v] = interpolate_image(
                        Psi_i[:, v].reshape(final_resolution),
                        xi[0, :], yi[:, 0], x[r][0, :], y[r][:, 0]
                    )

            execution_time += tm.time()-tic

            # Print scaling iteration info
            if print_info:
                print('Scaling iteration %d - ' % s + 'Resolution: ',
                      resolution[s], ', Area %f' % A[-1], file=print_file)

            tic = tm.time()

            # IMSA Initialization and Low-Order Reconstruction (page 4,
            # right column, 1) and 2-a topics)
            if s == 0:

                # Get empty arrays for particles and velocities
                output = initialize_arrays(P, N, R, V, contrast_type)

                # Contrast variables will be manipulated separately
                # from the electric field variables (x = {tau, Psi})
                tau, Psi, v_tau, v_psi, Phi = output[:5]

                # Personal best particles and evaluation
                t_tau, t_psi, Phi_t = output[5:8]

                # Global particle and evaluation
                g_tau, g_psi, Phi_g = output[8:11]

                # The variables are initialized according to 2-a in the
                # right column, page 4. Here, we are using the contrast
                # definition, i.e., not the complex permittivity value.
                if self.configuration.perfect_dielectric:
                    tau[0][:, :] = rnd.rand(P, N)
                elif self.configuration.good_conductor:
                    omega = 2*pi*self.configuration.f
                    tau[0][:, :] = -(rnd.rand(P, N)*self.configuration.sigma_b
                                     / (omega*self.configuration.epsilon_rb
                                        * epsilon_0))
                else:
                    omega = 2*pi*self.configuration.f
                    epsilon_rb = self.configuration.epsilon_rb
                    tau[0][:, :] = (rnd.rand(P, N)
                                    - 1j*(rnd.rand(P, N)
                                          * self.configuration.sigma_b
                                          / (omega*epsilon_rb*epsilon_0)))
                for p in range(P):
                    TAU = dia_matrix((tau[0][p, :].flatten(), 0), shape=(N, N))
                    Psi[0][p, :] = np.reshape(Psi_i_r[0] - Gint[0]@TAU@Psi_i_r[0],
                                              (1, -1))

            # High-order reconstruction
            else:

                # The paper does not explain in detail how particles from a
                # previous scale are represente in the next scaling iteration
                # Then, as we may assume that each new RoI comes from one
                # from the previous iteration, we interpolate the images.
                output = update_resolution(tau, Psi, v_tau, v_psi, t_tau,
                                           t_psi, g_tau, g_psi, last_x, last_y,
                                           last_region_ref, x, y)
                tau, Psi, v_tau, v_psi = output[:4]
                t_tau, t_psi = output[4:6]
                g_tau, g_psi = output[6:8]

            # Evaluate the particles. For the first iteration, it is
            # evaluation of the randomized initialization. For the
            # second and the following iterations, it is the evaluation
            # of the updated representation of solutions.
            self.__evaluate_particles(tau, Psi, Phi, Gext, Gint, Psi_s,
                                      Psi_i_r, number_evaluations, MAX_EVALS,
                                      max_evals_stop)
            if max_evals_stop:
                break

            if s > 0:
                # For the following iterations, it is necessary to
                # evaluate again the personal and global best solutions
                self.__evaluate_particles(t_tau, t_psi, Phi_t, Gext, Gint,
                                          Psi_s, Psi_i_r, number_evaluations,
                                          MAX_EVALS, max_evals_stop)
                if max_evals_stop:
                    break
                self.__evaluate_particles(g_tau, g_psi, Phi_g, Gext, Gint,
                                          Psi_s, Psi_i_r, number_evaluations,
                                          MAX_EVALS, max_evals_stop)
                if max_evals_stop:
                    break

            execution_time += tm.time()-tic

            # Run Particle Swarm Optimization
            out = self.__PSO(tau, Psi, v_tau, v_psi, t_tau, t_psi, g_tau,
                             g_psi, Phi, Phi_t, Phi_g, I[s], c1, c2, w, Gext,
                             Gint, Psi_s, Psi_i_r, x, y, xi, yi, Phi_th=Phi_th,
                             tau_min=tau_min, tau_max=tau_max,
                             convergence=convergence, miwi=MIWI,
                             number_evaluations=number_evaluations,
                             max_evaluations=MAX_EVALS, result=result,
                             instance=instance, print_info=print_info,
                             print_file=print_file)
            execution_time += out[1]

            tic = tm.time()

            if number_evaluations >= MAX_EVALS:
                max_evals_stop = True
                break

            # Multi-resolution process (s = 2, ..., S)
            if s != S-1:

                # The information of the coordinates of the current
                # iteration will be necessary for the next iteration
                # when we update the particles
                last_x, last_y = cp.deepcopy(x), cp.deepcopy(y)

                # Instead of linear indexation, it will be necessary
                # a matrix structure of the global best solution
                image = []
                for r in range(R):
                    image.append(g_tau[r].reshape(resolution[s]))

                # RoI updating
                p0, p1, last_region_ref = clustering(image, x, y)

                # Compute the area of new RoI
                A.append(0)
                for r in range(len(p0)):
                    A[-1] += (p1[r][0]-p0[r][0])*(p1[r][1]-p0[r][1])

                # Termination check (page 5, right-column, 3-b)
                if A[-1] == 0:
                    break
                elif (A[-1] - A[-2])/A[-1] < eta:
                    break

            execution_time += tm.time()-tic

        # When the algorithm is over, we need to transform the
        # information in the global best particle into an image of the
        # contrast map and the total electric field.
        self.__particle2results(g_tau, g_psi, Gext, x, y, xi, yi, result)
        result.execution_time = execution_time
        result.number_evaluations = number_evaluations
        result.objective_function = convergence.copy()

        return result, convergence

    def __particle2image(self, tau, Psi, xr, yr, xi, yi):
        """Convert the particle information into an image.

        Parameters
        ----------
            tau : list of :class:`numpy.ndarray`
                Contrast variables.

            psi : list of :class:`numpy.ndarray`
                Electric field variables.

            xr, yr : list of :class:`numpy.ndarray`
                Coordinates of RoI's.

            xi, yi : list of :class:`numpy.ndarray`
                Coordinates of the image domain.

        Returns
        -------
            epsilon_r, sigma : :class:`numpy.ndarray`
                Image of the relative permittivity and conductivity.
                Matrix shape.

            et : :class:`numpy.ndarray`
                Intern total electric field.
        """
        # Constants
        resolution = xr[0].shape
        R, N = len(xr), xr[0].size
        V = round(Psi[0].size/N)

        # Contrast and electric field variables
        if (self.configuration.perfect_dielectric
                or self.configuration.good_conductor):
            X = np.zeros(xi.shape)
        else:
            X = np.zeros(xi.shape, dtype=complex)
        et = np.zeros((xi.shape[0]*xi.shape[1], V), dtype=complex)

        # For each RoI
        for r in range(R):

            # The RoI map is interpolated at the coordinates of the
            # image to be recovered.
            aux = interpolate_image(tau[r].reshape(resolution), xr[r][0, :],
                                    yr[r][:, 0], xi[0, :], yi[:, 0],
                                    fill_value=0).reshape(xi.shape)

            # In order to address possible overlaping RoI's, the
            # contrast map is updated only where no contrast was added.
            X[X == 0] = aux[X == 0]

            aux_psi = Psi[r].reshape((N, V))
            aux = np.zeros((xi.size, V), dtype=complex)
            for v in range(V):
                aux[:, v] = interpolate_image(
                    aux_psi[:, v].reshape(xr[r].shape), xr[r][0, :],
                    yr[r][:, 0], xi[0, :], yi[:, 0], fill_value=0
                ).flatten()

            # The same is done for the electric field
            et[et == 0] = aux[et == 0]

        # Relative permittivity is only returned if the problem does not
        # assume good conductors.
        if (self.configuration.perfect_dielectric
                or not self.configuration.good_conductor):
            epsilon_r = cfg.get_relative_permittivity(
                X, self.configuration.epsilon_rb
            )
        else:
            epsilon_r = None

        # Conductivity is only returned if the problem does not assume
        # perfect dielectric objects.
        if self.configuration.good_conductor:
            sigma = cfg.get_conductivity(1j*X, 2*pi*self.configuration.f,
                                         self.configuration.epsilon_rb,
                                         self.configuration.sigma_b)
        elif not self.configuration.perfect_dielectric:
            sigma = cfg.get_conductivity(X, 2*pi*self.configuration.f,
                                         self.configuration.epsilon_rb,
                                         self.configuration.sigma_b)
        else:
            sigma = None

        return epsilon_r, sigma, et

    def __PSO(self, tau, Psi, v_tau, v_psi, t_tau, t_psi, g_tau, g_psi, Phi,
              Phi_t, Phi_g, I, c1, c2, w, Gext, Gint, Psi_s, Psi_i, xr, yr, xi,
              yi, Phi_th=-np.inf, tau_min=None, tau_max=None,
              max_evaluations=np.inf, number_evaluations=0, miwi=np.inf,
              result=None, instance=None, convergence=[], print_info=True,
              print_file=sys.stdout):
        """Run Particle Swarm Optimization.

        Parameters
        ----------
            tau, Psi : list of :class:`numpy.ndarray`
                List of contrast and electric field particles per RoI.
                They must be initialized out of the algorithm

            v_tau, v_psi : list of :class:`numpy.ndarray`
                List of contrast and electric field velocities per RoI.

            t_tau, t_psi : list of :class:`numpy.ndarray`
                List of personal best particles (contrast and electric
                field) per RoI.

            g_tau, g_psi : list of :class:`numpy.ndarray`
                List of global best particle (contrast and electric
                field) per RoI.

            Phi, Phi_t : :class:`numpy.ndarray`
                Cost function evaluation of common and personal best
                particles.

            Phi_g : float
                Cost function evaluation of global best particle.

            I : int
                Number of iterations

            c1, c2, w : float
                Acceleration coefficients and inertial weight parameters
                for velocity update.

            Gext, Gint : list of :class:`numpy.ndarray`
                Green function matrix for each RoI.

            Psi_s, Psi_i : list of :class:`numpy.ndarray`
                Scattered field data and incident field per RoI.

            xr, yr : list of :class:`numpy.ndarray`
                Coordinates of each element for each RoI.

            xi, yi : :class:`numpy.ndarray`
                Coordinates of the image to be recovered as a whole.

            Phi_th : float, default: `numpy.inf`
                Threshold value for cost function of the global best
                particle for stopping the algorithm.

            tau_min, tau_max : float or complex, default: None
                Minimum and maximum values for contrast variables.

            max_evaluations : int, default: `numpy.inf`
                Maximum number of evaluations for stopping the
                algorithm.

            number_evaluations : int, default: 0
                Evaluations counter.

            miwi : int, default: `numpy.inf`
                Maximum number of iterations without improvement.

            result : :class:`results.Results`, default: None
                Result object.

            instance : :class:`inputdata.InputData`, default: None
                Instance object which is being solved.

            convergence : list, default: []
                Record of cost function evaluation of the global best
                particle.

            print_info : bool, default: True
                Print or not information.

            print_file, default: `sys.stdout`
                An object with a write method for printing information.

        Returns
        -------
            number_evaluations : int

            execution_time : float
                Time elapsed.
        """
        # Constants
        R, P, N = len(tau), tau[0].shape[0], tau[0].shape[1]
        V = round(Psi[0].shape[1]/N)

        # Auxiliar variables
        perfect_dielectric = self.configuration.perfect_dielectric
        good_conductor = self.configuration.good_conductor
        no_improvement_counter = 0
        last_Phi_g = None

        # Time elapsed
        execution_time = 0.

        for i in range(I):

            tic = tm.time()

            # Update personal best particles
            for p in range(P):
                if Phi[p] < Phi_t[p]:
                    for r in range(R):
                        t_tau[r][p, :] = tau[r][p, :]
                        t_psi[r][p, :] = Psi[r][p, :]
                    Phi_t[p] = Phi[p]

            # Update global best particle
            last_Phi_g = Phi_g
            if np.amin(Phi_t) < Phi_g:
                p = np.argsort(Phi_t)[0]
                for r in range(R):
                    g_tau[r][:] = t_tau[r][p, :]
                    g_psi[r][:] = t_psi[r][p, :]
                Phi_g = Phi_t[p]

            # Update counter
            if i > 0 and np.abs((Phi_g-last_Phi_g)/last_Phi_g) < 1e-4:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            execution_time += tm.time()-tic

            # Updating error measures
            solution = self.__particle2results(g_tau, g_psi, Gext,
                                               xr, yr, xi, yi)

            result.update_error(instance, scattered_field=solution.es,
                                total_field=solution.et,
                                relative_permittivity_map=solution.epsilon_r,
                                conductivity_map=solution.sigma)

            # Stop criterion
            if Phi_g < Phi_th:
                break
            if no_improvement_counter >= miwi:
                break

            tic = tm.time()

            # Update particle velocity
            for r in range(R):
                v_tau[r] = (w*v_tau[r]
                            + c1*rnd.rand(P, N)*(t_tau[r]-tau[r])
                            + c2*rnd.rand(P, N)*(np.tile(g_tau[r], (P, 1))
                                                 - tau[r]))
                v_psi[r] = (w*v_psi[r]
                            + c1*rnd.rand(P, N*V)*(t_psi[r]-Psi[r])
                            + c2*rnd.rand(P, N*V)*(np.tile(g_psi[r], (P, 1))
                                                   - Psi[r]))

                # Velocity variables are clamped (page 5, left column,
                # footnote 4)
                if tau_min is not None and tau_max is not None:
                    if perfect_dielectric:
                        v_tau[r][v_tau[r] > np.real(tau_max-tau_min)] = (
                            np.real(tau_max-tau_min)
                        )
                    elif good_conductor:
                        v_tau[r][v_tau[r] > np.imag(tau_max-tau_min)] = (
                            np.imag(tau_max-tau_min)
                        )
                    else:
                        idx = np.real(v_tau[r]) > np.real(tau_max-tau_min)
                        v_tau[r][idx] = np.real(tau_max-tau_min)
                        idx = np.imag(v_tau[r]) > np.imag(tau_max-tau_min)
                        v_tau[r][idx] = np.imag(tau_max-tau_min)

            # Update particle position
            for r in range(R):
                tau[r] = tau[r] + v_tau[r]
                Psi[r] = Psi[r] + v_psi[r]

                # Reflecting wall boundary conditions for the contrast
                # values. The same boundary condition was not
                # implemented for electric field variables since the
                # authors did not provide neither values nor strategy
                # to define this kind of bounds.
                if tau_min is not None:
                    if perfect_dielectric:
                        idx = tau[r] < np.real(tau_min)
                        tau[r][idx] = 2*np.real(tau_min) - tau[r][idx]
                        # tau[r][idx] = np.real(tau_min)
                    elif good_conductor:
                        idx = tau[r] < np.imag(tau_min)
                        tau[r][idx] = 2*np.imag(tau_min) - tau[r][idx]
                    else:
                        idx = np.real(tau[r]) < np.real(tau_min)
                        tau[r][idx] = 2*np.real(tau_min)-np.real(tau[r][idx])
                        idx = np.imag(tau[r]) < np.imag(tau_min)
                        tau[r][idx] = 2*np.imag(tau_min)-np.imag(tau[r][idx])
                if tau_max is not None:
                    if perfect_dielectric:
                        idx = tau[r] > np.real(tau_max)
                        tau[r][idx] = 2*np.real(tau_max) - tau[r][idx]
                    elif good_conductor:
                        idx = tau[r] > np.imag(tau_max)
                        tau[r][idx] = 2*np.imag(tau_max) - tau[r][idx]
                    else:
                        idx = np.real(tau[r]) > np.real(tau_max)
                        tau[r][idx] = 2*np.real(tau_max)-np.real(tau[r][idx])
                        idx = np.imag(tau[r]) > np.imag(tau_max)
                        tau[r][idx] = 2*np.imag(tau_max)-np.imag(tau[r][idx])

                if tau_min is not None:
                    tau[r][np.real(tau[r]) < np.real(tau_min)] = (
                        np.real(tau_min)
                    )
                    tau[r][np.imag(tau[r]) < np.imag(tau_min)] = (
                        np.imag(tau_min)
                    )

            # Update particle evaluation
            self.__evaluate_particles(tau, Psi, Phi, Gext, Gint, Psi_s, Psi_i,
                                      number_evaluations, max_evaluations,
                                      None)

            if number_evaluations >= max_evaluations:
                break

            # Record best solution fitness
            convergence.append(Phi_g)
            if number_evaluations == max_evaluations:
                break

            execution_time += tm.time()-tic

            if print_info and I <= 10:
                print('  PSO Iteration %d' % i
                      + ' - Best Phi(x): %.3e' % Phi_g,
                      file=print_file)
            elif print_info and i % (I/10) == 0:
                print('  PSO Iteration %d' % i
                      + ' - Best Phi(x): %.3e' % Phi_g,
                      file=print_file)
            elif print_info and i == I-1:
                print('  PSO Iteration %d' % i
                      + ' - Best Phi(x): %.3e' % Phi_g,
                      file=print_file)

        return number_evaluations, execution_time

    def __get_values(self, measure, executions):
        """Return error measures from executions.

        Parameters
        ----------
            measure : str
                Name of the meausure. Options: zeta_rn, zeta_rpad,
                zeta_epad, zeta_ebe, zeta_eoe, zeta_sad, zeta_sbe,
                zeta_soe, zeta_be, zeta_tfmpad, zeta_tfppad.

            executions : list of :class:`results.Results`
                List with the results of each execution.

        Returns
        -------
            values : list of list
                Each element of the main list contains a list with the
                error value per iteration.
        """
        values = []
        for ne in range(len(executions)):
            if measure == 'zeta_rn':
                values.append(executions[ne].zeta_rn)
            elif measure == 'zeta_rpad':
                values.append(executions[ne].zeta_rpad)
            elif measure == 'zeta_epad':
                values.append(executions[ne].zeta_epad)
            elif measure == 'zeta_ebe':
                values.append(executions[ne].zeta_ebe)
            elif measure == 'zeta_eoe':
                values.append(executions[ne].zeta_eoe)
            elif measure == 'zeta_sad':
                values.append(executions[ne].zeta_sad)
            elif measure == 'zeta_sbe':
                values.append(executions[ne].zeta_sbe)
            elif measure == 'zeta_soe':
                values.append(executions[ne].zeta_soe)
            elif measure == 'zeta_be':
                values.append(executions[ne].zeta_be)
            elif measure == 'zeta_tfmpad':
                values.append(executions[ne].zeta_tfmpad)
            elif measure == 'zeta_tfppad':
                values.append(executions[ne].zeta_tfppad)
        return values

    def __get_means(self, values, percent):
        """Compute means per percent of iterations.

        Parameters
        ----------
            values : list of list
                Each element of the main list has a list with the error
                value for each iteration.

            percent : :class:`numpy.ndarray`
                Percentages of the number of iterations.

        Returns
        -------
            means : :class:`numpy.ndarray`
                Mean in each percentage of iterations.
        """
        nmeans = percent.size
        means = np.zeros((len(values), nmeans))
        for i in range(len(values)):
            nsamples = len(values[i])
            idx = np.round(percent/100*(nsamples-1)).astype(int)
            for j in range(idx.size):
                means[i, j] = values[i][idx[j]]
        return np.mean(means, axis=0).tolist()

    def __evaluate_particles(self, tau, Psi, Phi, Gext, Gint, Psi_s, Psi_i,
                             number_evaluations, MAX_EVALS, max_evals_stop):
        """Evaluate the cost function for each particle.

        Parameters
        ----------
            tau, Psi: list of :class:`numpy.ndarray`
                List of contrast and electric field variables.

            Phi : :class:`numpy.ndarray`
                Array to solve the evaluation.

            Gext, Gint : list of :class:`numpy.ndarray`
                Green function matrix for each RoI.

            Psi_s, Psi_i : list of :class:`numpy.ndarray`
                Scattered field data and incident field per RoI.

            number_evaluations : int
                Evaluations counter.

            MAX_EVALS : int
                Maximum number of evaluations for stopping the
                algorithm.

            max_evals_stop : bool
                Flag to indicate if the maximum number of evaluations
                was reached.
        """
        R = len(tau)

        # Multiples particles
        if tau[0].ndim == 2:
            P, N = tau[0].shape
            V = round(Psi[0].shape[1]/N)
            single_particle = False

        # Single particle
        else:
            P, N = 1, tau[0].size
            V = round(Psi[0].size/N)
            single_particle = True

        for p in range(P):
            TAU, PSI = [], []

            for r in range(R):

                # Multiples particles
                if single_particle:
                    aux1 = tau[r][:]
                    aux2 = Psi[r][:]

                # Single particle
                else:
                    aux1 = tau[r][p, :]
                    aux2 = Psi[r][p, :]

                if self.configuration.good_conductor:
                    TAU.append(dia_matrix((1j*aux1.flatten(), 0),
                                          shape=(N, N)))
                else:
                    TAU.append(dia_matrix((aux1.flatten(), 0), shape=(N, N)))
                PSI.append(aux2.reshape((N, V)))

            if single_particle:
                Phi = Phi_eval(TAU, PSI, Gext, Gint, Psi_s, Psi_i)
            else:
                Phi[p] = Phi_eval(TAU, PSI, Gext, Gint, Psi_s, Psi_i)

            number_evaluations += 1
            if number_evaluations == MAX_EVALS:
                max_evals_stop = True
                break

    def __particle2results(self, tau, Psi, Gext, x, y, xi, yi, result=None):
        """Convert the information from the particle to a result object.

        Parameters
        ----------
            tau, Psi : list of :class:`numpy.ndarray`
                Contrast and electric field variables of each RoI.

            Gext : list of :class:`numpy.ndarray`
                Green function matrix for each RoI.

            x, y : list of :class:`numpy.ndarray`
                Coordinates of each element for each RoI.

            xi, yi : :class:`numpy.ndarray`
                Coordinates of the image to be recovered as a whole.

            result : :class:`results.Results`, default: None
                Result object. If none is provided, then it will be
                returned a new one.

        Returns
        -------
            result : :class:`results.Results`
        """
        if result is None:
            result = rst.Results(name='')

        # Constants
        R, N = len(x), x[0].size
        V = round(Psi[0].size/N)

        # Get contrast and eletric field maps
        out = self.__particle2image(tau, Psi, x, y, xi, yi)
        result.epsilon_r = out[0]
        result.sigma = out[1]
        result.et = out[2]

        # Compute scattered field
        TAU, PSI = [], []
        for r in range(R):
            if (tau[r].dtype == complex
                    or self.configuration.perfect_dielectric):
                TAU.append(dia_matrix((tau[r][:].flatten(), 0),
                                      shape=(N, N)))
            elif self.configuration.good_conductor:
                TAU.append(dia_matrix((1j*tau[r][:].flatten(), 0),
                                      shape=(N, N)))
            PSI.append(Psi[r][:].reshape(N, V))
        result.es = Psi_s_eval(TAU, PSI, Gext)

        return result

    def __execution_info(self, number_executions, min_contrast, max_contrast,
                         store_image, percent_step, save_executions):
        """Return string with execution information."""
        text = 'Number of particles (P): %d' % self.number_particles
        text += '\nc1 = %.1f, ' % self.c1 + 'c2 = %.1f, ' % self.c2
        text += 'w = %.1f' % self.w
        text += '\nScaling steps (S): %d' % len(self.resolution)

        if all(self.resolution[i] == self.resolution[0]
               for i in range(1, len(self.resolution))):
            text += '\nResolution steps: ' + str(self.resolution[0])
        else:
            text += '\nResolution steps: ' + str(self.resolution)

        if all(self.number_iterations[i] == self.number_iterations[0]
               for i in range(1, len(self.number_iterations))):
            text += '\nNumber of iterations: %d' % self.number_iterations[0]
        else:
            text += '\nNumber of iterations: ' + str(self.number_iterations)

        if self.Phi_th != -np.inf and self.Phi_th is not None:
            text += '\nCost function threshold: %.3e' % self.Phi_th

        if self.eta != -np.inf and self.eta is not None:
            text += '\nArea reduction threshold : %.2f' % self.eta

        if (self.maximum_evaluations != np.inf
                and self.maximum_evaluations is not None):
            text += ('\nMaximum number of evaluations: %d'
                     % self.maximum_evaluations)

        text += '\nNumber of executions: %d' % number_executions

        if min_contrast is not None:
            text += '\nMinimum contrast allowed: %.3e' % min_contrast

        if max_contrast is not None:
            text += '\nMaximum contrast allowed: %.3e' % max_contrast

        text += '\nStoring the ' + store_image + ' recovered image'
        text += '\nPercent step for convergence sampling: %.1f' % percent_step

        if save_executions is not None:
            text += '\nSave executions in file: ' + save_executions

        return text


def green_function(xm, ym, x, y, cell_area, kb, resolution):
    """Compute Green function matrix [1]_.

    Parameters
    ----------
        xm, ym : :class:`numpy.ndarray`
            Arrays with the cartesian coordinates of the measurement
            points.

        x, y : list of :class:`numpy.ndarray`
            List with the cartesian coordinates of each RoI.

        cell_area : :class:`numpy.ndarray`
            Array with the cell area of each RoI.

        kb : float or complex
            Wavenumber.

        resolution : tuple
            Image size in pixels for all RoI's.

    Returns
    -------
        Gext, Gint : list of :class:`numpy.ndarray`
            List of Green function matrices for data and state
            equations. Nomenclature according to [2]_.

    References
    ----------
        .. [1] Richmond, Jack. "Scattering by a dielectric cylinder of
           arbitrary cross section shape." IEEE Transactions on Antennas
           and Propagation 13.3 (1965): 334-341.
        .. [2] Salucci, Marco, et al. "Multifrequency particle swarm
           optimization for enhanced multiresolution GPR microwave
           imaging." IEEE Transactions on Geoscience and Remote Sensing
           55.3 (2016): 1305-1317.
    """
    M, R = xm.size, len(x)

    # Data equation
    Gext = []
    for r in range(R):
        N = x[r].size
        Gext.append(np.zeros((M, N), dtype=complex))
        for m in range(M):
            rho = np.sqrt((x[r]-xm[m])**2 + (y[r]-ym[m])**2).flatten()
            Gext[r][m, :] = (1j*pi*kb*cell_area[r]/2*jv(1, kb*cell_area[r])
                             * hankel2(0, kb*rho))

    # State Equation
    Gint = []
    for r in range(R):
        N = x[r].size
        Gint.append(np.zeros((N, N), dtype=complex))
        for n in range(N):
            i, j = np.unravel_index(n, resolution)
            rho = np.sqrt((x[r]-x[r][i, j])**2
                          + (y[r]-y[r][i, j])**2).flatten()
            Gint[r][n, :] = (1j*pi*kb*cell_area[r]/2*jv(1, kb*cell_area[r])
                             * hankel2(0, kb*rho))
            Gint[r][n, rho == 0] = 1j/2*(pi*kb*cell_area[r]
                                       * hankel2(1, kb*cell_area[r])-2j)

    return Gext, Gint


def initialize_arrays(P, N, R, V, contrast_type=float):
    """Allocate arrays for particles and velocities.

    Parameters
    ----------
        P : int
            Number of particles.

        N : int
            Number of variables.

        R : int
            Number of RoI's.

        V : int
            Number of incidences.

        contrast_type : float or complex, default: float
            Type of contrast variables. Float for perfect dielectric or
            good conductors objects. Complex otherwise.

    Returns
    -------
        tau, Psi : list of :class:`numpy.ndarray`
            Contrast and electric field variables for each RoI. Each
            element of the list is a single RoI with a matrix in which
            the rows are the particles and the columns are the
            variables.

        v_tau, v_psi : list of :class:`numpy.ndarray`
            Velocity arrays with the same structure than tau and Psi.

        Phi : :class:`numpy.ndarray`
            Array for recording the cost function evaluation of the
            particles.

        t_tau, t_psi : list of :class:`numpy.ndarray`
            Personal best particles for contrast and electric field.

        Phi_t : :class:`numpy.ndarray`
            Array for recording the cost function evaluation of the
            personal best particles.

        g_tau, g_psi : list of :class:`numpy.ndarray`
            Global best particle for contrast and electric field.

        Phi_g : float
            Cost function evaluation of the global best particle.
    """
    # Main particles
    tau, Psi, v_tau, v_psi = [], [], [], []
    for i in range(R):
        tau.append(np.zeros((P, N), dtype=contrast_type))
        Psi.append(np.zeros((P, N*V), dtype=complex))
        v_tau.append(np.zeros((P, N), dtype=contrast_type))
        v_psi.append(np.zeros((P, N*V), dtype=complex))
    Phi = np.zeros(P)

    # Personal best particles
    t_tau, t_psi = [], []
    for i in range(R):
        t_tau.append(np.zeros((P, N), dtype=contrast_type))
        t_psi.append(np.zeros((P, N*V), dtype=complex))
    Phi_t = np.inf*np.ones(Phi.shape)

    # Global best particles
    g_tau, g_psi = [], []
    for i in range(R):
        g_tau.append(np.zeros(N, dtype=contrast_type))
        g_psi.append(np.zeros(N*V, dtype=complex))
    Phi_g = np.inf

    return (tau, Psi, v_tau, v_psi, Phi,
            t_tau, t_psi, Phi_t,
            g_tau, g_psi, Phi_g)


def update_resolution(tau, Psi, v_tau, v_psi, t_tau, t_psi, g_tau, g_psi,
                      x, y, region_ref, new_x, new_y):
    """Update the particles for a new set of RoI's.

    Each particle is updated as follows: first we identify which old
    RoI the new one belongs to and then we interpolate the values.
    Therefore, the particles variables are reshaped to the matrix
    representation and then it is interpolated.

    Parameters
    ----------
        tau, Psi : list of :class:`numpy.ndarray`
            Contrast and electric variables to be updated.

        v_tau, v_psi : list of :class:`numpy.ndarray`
            Velocity arrays to be updated.

        t_tau, t_psi : list of :class:`numpy.ndarray`
            Personal best particles to be updated.

        g_tau, g_psi : list of :class:`numpy.ndarray`
            Global best particle to be updated.

        x, y : list of :class:`numpy.ndarray`
            Cartesian coordinates of each current RoI. They represent
            cartesian coordinates of the variables in the given
            particles.

        region_ref : :class:`numpy.ndarray`
            Array with the size equals to the number of new RoI's in
            which each element is the index of the current RoI which
            contains the new RoI. For example, the i-th new RoI
            indicated in the i-th element of `new_x` and `new_y` belongs
            to RoI which is indicated in the i-th position of
            `region_ref`, considering the order in `x` and `y`.
            Therefore, this variables indicates which current RoI the
            new one belongs to.

        new_x, new_y : list of :class:`numpy.ndarray`
            Cartesian coordinates of the new RoI's.

    Returns
    -------
        new_tau, new_Psi : list of :class:`numpy.ndarray`
            The contrast and electric field variables updated to the
            new RoI's.

        new_v_tau, new_v_psi : list of :class:`numpy.ndarray`
            The contrast and electric field velocities updated to the
            new RoI's.

        new_t_tau, new_t_psi : list of :class:`numpy.ndarray`
            The contrast and electric field variables of personal best
            particles updated to the new RoI's.

        new_g_tau, new_g_psi : list of :class:`numpy.ndarray`
            The contrast and electric field variables of the global best
            particle updated to the new RoI's.
    """
    # Arrays shapes
    P, R, NO, NN = tau[0].shape[0], len(new_x), tau[0].shape[1], new_x[0].size
    V = round(Psi[0].shape[1]/NO)

    # Allocate new arrays
    output = initialize_arrays(P, NN, R, V, tau[0].dtype)
    new_tau, new_Psi, new_v_tau, new_v_psi = output[:4]
    new_t_tau, new_t_psi = output[5:7]
    new_g_tau, new_g_psi = output[8:10]

    # For each particle
    for p in range(P):

        # For each new RoI
        for r in range(R):

            # Identify which old RoI the new one belongs to
            k = region_ref[r]

            # Record coordinates in auxiliar variables (uniform mesh)
            ox, oy = x[k][0, :], y[k][:, 0]
            old_resolution = np.shape(x[k])

            # Interpolate contrast map
            new_tau[r][p, :] = interpolate_image(
                tau[k][p, :].reshape(old_resolution),
                ox, oy, new_x[r][0, :], new_y[r][:, 0]
            )

            # Interpolate contrast velocity
            new_v_tau[r][p, :] = interpolate_image(
                v_tau[k][p, :].reshape(old_resolution),
                ox, oy, new_x[r][0, :], new_y[r][:, 0]
            )

            # Interpolate contrast map of personal best particle
            new_t_tau[r][p, :] = interpolate_image(
                t_tau[k][p, :].reshape(old_resolution),
                ox, oy, new_x[r][r, :], new_y[r][:, 0]
            )

            # Interpolate the electric field maps (each incidence is a
            # different map).
            aux1 = Psi[k][p, :].reshape((NO, V))
            aux2 = np.zeros((NN, V), dtype=complex)
            for v in range(V):
                aux2[:, v] = interpolate_image(
                    aux1[:, v].reshape(old_resolution), ox, oy,
                    new_x[r][0, :], new_y[r][:, 0]
                ).flatten()
            new_Psi[r][p, :] = aux2.flatten()

            # Interpolate the electric field velocity (each incidence is
            # a different map).
            aux1 = v_psi[k][p, :].reshape((NO, V))
            aux2 = np.zeros((NN, V), dtype=complex)
            for v in range(V):
                aux2[:, v] = interpolate_image(
                    aux1[:, v].reshape(old_resolution), ox, oy,
                    new_x[r][0, :], new_y[r][:, 0]
                ).flatten()
            new_v_psi[r][p, :] = aux2.flatten()

            # Interpolate the electric field maps of personal best
            # particle (each incidence is a different map).
            aux1 = t_psi[k][p, :].reshape((NO, V))
            aux2 = np.zeros((NN, V), dtype=complex)
            for v in range(V):
                aux2[:, v] = interpolate_image(
                    aux1[:, v].reshape(old_resolution), ox, oy,
                    new_x[r][0, :], new_y[r][:, 0]
                ).flatten()
            new_t_psi[r][p, :] = aux2.flatten()

    # Update global best particle
    for r in range(R):

        # Identify which old RoI the new one belongs to
        k = region_ref[r]

        # Record coordinates in auxiliar variables (uniform mesh)
        ox, oy = x[k][0, :], y[k][:, 0]
        old_resolution = np.shape(x[k])

        # Interpolate contrast map
        new_g_tau[r][:] = interpolate_image(g_tau[k].reshape(old_resolution),
                                            ox, oy, new_x[r][0, :],
                                            new_y[r][:, 0])

        # Interpolate the electric field maps (each incidence is a
        # different map).
        aux1 = g_psi[k][:].reshape((NO, V))
        aux2 = np.zeros((NN, V), dtype=complex)
        for v in range(V):
            aux2[:, v] = interpolate_image(
                aux1[:, v].reshape(old_resolution), ox, oy,
                new_x[r][0, :], new_y[r][:, 0]
            ).flatten()
        new_g_psi[r][:] = aux2.flatten()

    return (new_tau, new_Psi, new_v_tau, new_v_psi,
            new_t_tau, new_t_psi, new_g_tau, new_g_psi)


def interpolate_image(old_image, old_x, old_y, new_x, new_y, fill_value=None):
    """Interpolate an image.

    Parameters
    ----------
        old_image : :class:`numpy.ndarray`
            Image to be interpolated in matrix shape (meshgrid format).

        old_x, old_y : :class:`numpy.ndarray`
            1D arrays to indicate the cartesian coordinates of the image
            to be interpolated.

        new_x, new_y : :class:`numpy.ndarray`
            1D arrays to indicate the cartesian coordinates in which
            `old_image` will be interpolated.

        fill_value : float, default: None
            If provided, the value to use for points outside of the
            interpolation domain. If omitted (None), values outside the
            domain are extrapolated via nearest-neighbor extrapolation.

    Returns
    -------
        new_image : :class:`numpy.ndarray`
            Interpolated image in flatten shape.
    """
    # For real values
    if old_image.dtype == float:
        f = interp2d(old_x, old_y, old_image, fill_value=fill_value)
        new_image = f(new_x, new_y).flatten()

    # For complex values
    elif old_image.dtype == complex:
        fr = interp2d(old_x, old_y, np.real(old_image), fill_value=fill_value)
        fi = interp2d(old_x, old_y, np.imag(old_image), fill_value=fill_value)
        new_image = (fr(new_x, new_y).flatten()
                     + 1j*fi(new_x, new_y).flatten())

    return new_image


def Phi_eval(tau, Psi, Gext, Gint, Psi_s, Psi_i):
    r"""Evaluate the cost function [1]_.

    The cost function is the weighted sum of residuals from the data
    and state equations. The weights are the sum of scattered and
    incident fields, respectively.

    .. math::

        \Phi(\tau, \Psi) = \frac{\sum_{v=1}^V\sum_{m=1}^M
        |\Psi_{S}^{m,v} + \mathbf{G}_{ext}^{m}\boldsymbol{\tau}
        \boldsymbol{\Psi}^v|^2}{\sum_{v=1}^V\sum_{m=1}^M
        |\Psi_S^{m,v}|^2} + \frac{\sum_{v=1}^V\sum_{n=1}^N \Psi_i^{n,v}
        - \Psi^{n,v} - \mathbf{G}_{int}^{n}\boldsymbol{\tau}
        \boldsymbol{\Psi}^v} {\sum_{v=1}^V\sum_{n=1}^N |\Psi_i^{n,v}|^2}

    Parameters
    ----------
        tau : list :class:`scipy.sparse.dia_matrix`
            List of sparse diagonal matrix with the contrast values in
            the main diagonal for each RoI.

        Psi : list of :class:`numpy.ndarray`
            List of intern total electric field matrices. Each matrix is
            the electric field for a RoI (shape: pixels x incidence).

        Gext, Gint : list of :class:`numpy.ndarray`
            List of external and internal Green function matrices for
            each RoI (shape: measurement x pixels).

        Psi_s : :class:`numpy.ndarray`
            Scattered field data (shape: measureament x incidence).

        Psi_i : list of :class:`numpy.ndarray`
            Incident field for each RoI (shape: pixels x incidence).

    Returns
    -------
        fx : float
            Cost function evaluation.

    References
    ----------
        .. [1] Salucci, Marco, et al. "Multifrequency particle swarm
           optimization for enhanced multiresolution GPR microwave
           imaging." IEEE Transactions on Geoscience and Remote Sensing
           55.3 (2016): 1305-1317.
    """
    # Number of RoI's
    R = len(tau)

    # Residual of data equation
    aux_data = np.zeros(Psi_s.shape, dtype=complex)
    for r in range(R):
        aux_data += Gext[r] @ tau[r] @ Psi[r]

    if np.amin(np.abs(Psi_s + aux_data)) > 1e20:
        for r in range(R):
            print('max of tau[r] = %.3e' % np.amax(np.abs(tau[r].toarray())))
            print('min of Psi[r] = %.3e' % np.amax(np.abs(Psi[r])))
        exit()
    fx = np.sum(np.abs(Psi_s + aux_data)**2)/np.sum(np.abs(Psi_s)**2)

    # Residual of state equation
    aux_state = np.zeros((2, R))
    for r in range(R):
        aux_state[0, r] = np.sum(np.abs(Psi_i[r] - Psi[r]
                                        - Gint[r] @ tau[r] @ Psi[r])**2)
        aux_state[1, r] = np.sum(np.abs(Psi_i[r])**2)
    fx += np.sum(aux_state[0, :])/np.sum(aux_state[1, :])

    # for r in range(R):
    #     fx += 1e0*np.sum(np.abs(tau[r])**2)

    return fx


def Psi_s_eval(tau, Psi, Gext):
    """Compute integral equation to estimate scattered field.

    Parameters
    ----------
        tau : list :class:`scipy.sparse.dia_matrix`
            List of sparse diagonal matrix with the contrast values in
            the main diagonal for each RoI.

        Psi : list of :class:`numpy.ndarray`
            List of intern total electric field matrices. Each matrix is
            the electric field for a RoI (shape: pixels x incidence).

        Gext : list of :class:`numpy.ndarray`
            List of external Green function matrices for each RoI
            (shape: measurement x pixels).

    Returns
    -------
        Psi_s : :class:`numpy.ndarray`
            Scattered field matrix (shape: measurement x incidence).
    """
    Psi_s = -Gext[0] @ tau[0] @ Psi[0]
    for r in range(1, len(Gext)):
        Psi_s += -Gext[r] @ tau[r] @ Psi[r]
    return Psi_s


def clustering(tau, x, y):
    """Determine new RoI's.

    According to [1]_:

        *"The clustering procedure aimed at defining the number
        :math:`Q` of scatterers in the investigation domain and the
        regions :math:`D_O^{(q)}, q = 1, ..., Q` where the synthetic
        zoom will be performed."*

    This procedure consists in 4 steps: thresholding, noise filtering,
    object detection and centroid and side computation [2]_.

    Parameters
    ----------
        tau : list of :class:`numpy.ndarray`
            List of contrast images.

        x, y : list of :class:`numpy.ndarray`
            List of cartesian coordinates for each current RoI.

    Returns
    -------
        p0, p1 : list of tuples
            List with the left-lower and right-upper points of the new
            RoI's. Each point is a tuple in (x, y) format.

        ipar : list of int
            Reference list to indicate which old RoI a new one belongs
            to. So, for example, the i-th element of the list means
            the index of the old RoI for which the new i-th RoI belongs
            to.

    References
    ----------
    .. [1] Caorsi, Salvatore, et al. "A new methodology based on an
       iterative multiscaling for microwave imaging." IEEE transactions
       on microwave theory and techniques 51.4 (2003): 1162-1173.
    .. [2] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa.
       "Detection, location, and imaging of multiple scatterers by means
       of the iterative multiscaling method." IEEE transactions on
       microwave theory and techniques 52.4 (2004): 1217-1228.
    """
    # Apply threshold
    binary = thresholding(tau)

    # Filter the binary image
    filtering(binary)

    # Detect objects/areas to zoom
    labels, ipar = object_detecting(binary)
    number_of_areas = len(ipar)

    xc, yc, L = new_areas(labels, binary, len(ipar), x, y)

    # Compute the limits of the objects areas
    if number_of_areas > 0 and not np.all(L == 0):

        # Define regions
        p0, p1 = [], []
        new_ipar = []
        for i in range(len(xc)):
            if L[i] != 0:
                p0.append((xc[i]-L[i]/2, yc[i]-L[i]/2))
                p1.append((xc[i]+L[i]/2, yc[i]+L[i]/2))
                new_ipar.append(ipar[i])
        ipar = new_ipar.copy()

    # If no object is detected
    else:
        p0, p1, ipar = [], [], []
        for i in range(len(x)):
            p0.append((x[i][0, 0], y[i][0, 0]))
            p1.append((x[i][-1, -1], y[i][-1, -1]))
            ipar.append(i)

    return p0, p1, ipar


def compute_mesh(p0, p1, resolution):
    """Get mesh for a given set of areas and resolution.

    Parameters
    ----------
        p0, p1 : list of tuples
            List with the left-lower and right-upper points of the new
            RoI's. Each point is a tuple in (x, y) format.

        resolution : tuple
            Resolution of meshes. Must be in (NY, NX) format.

    Returns
    -------
        x, y : list of :class:`numpy.ndarray`
            List of meshgrids.

        cell_area : :class:`numpy.ndarray`
            1D array with the cell area for each mesh according to [1]_.

    References
    ----------
        .. [1] Pastorino, Matteo. Microwave imaging. Vol. 208. John
           Wiley & Sons, 2010.
    """
    # Number of RoI's
    R = len(p0)

    # Number of pixels
    N = resolution[0]*resolution[1]

    # Allocate list and array
    x, y = [], []
    cell_area = np.zeros(R)

    # For each RoI
    for r in range(R):

        # Side length
        Lx, Ly = p1[r][0]-p0[r][0], p1[r][1]-p0[r][1]

        # Cell size
        dy, dx = Ly/resolution[0], Lx/resolution[1]

        # Coordinates in meshgrid format
        auxx, auxy = cfg.get_coordinates_ddomain(dx=dx, dy=dy, xmin=p0[r][0],
                                                 xmax=p1[r][0], ymin=p0[r][1],
                                                 ymax=p1[r][1])
        x.append(np.copy(auxx))
        y.append(np.copy(auxy))
        cell_area[r] = np.sqrt(dx*dy/pi)

    return x, y, cell_area


def neighbors(i, j, I, J, neighbors=8):
    """Get a list with the indexes of neighbors cells.

    Parameters
    ----------
        i, j : int
            Row and column indexes (or the opposite) of the center cell.

        I, J : int
            Number of rows and columns (or the opposite).

        neighbors : {4, 8}, default: 8
            Amount of neighbors. If 4, then the east, west, north and
            soulth neighbors are considered. If 8, then the northwest,
            northeast, soulthwest and soultheast are added.

    Returns
    -------
        ilist, jlist : list
    """
    if neighbors == 4:
        possible_i = [i-1, i+1,   i,   i]
        possible_j = [j,     j, j-1, j+1]
    elif neighbors == 8:
        possible_i = [i-1, i-1, i-1,   i,   i, i+1, i+1, i+1]
        possible_j = [j-1,   j, j+1, j-1, j+1, j-1,   j, j+1]

    ilist, jlist = [], []
    for n in range(len(possible_i)):
        if (possible_i[n] < 0 or possible_i[n] == I
                or possible_j[n] < 0 or possible_j[n] == J):
            pass
        else:
            ilist.append(possible_i[n])
            jlist.append(possible_j[n])

    return ilist, jlist


def thresholding(image):
    """Binarization by thresholding.

    Thresholding procces to transform the image into a binary one. The
    threshold value `T` is computed by the minimum bin of the
    histogram of the image [1]_.

    Parameters
    ----------
        image : :class:`numpy.ndarray` or list
            List of image (or a single one) to be binarized.

    Returns
    -------
        binary : list of :class:`numpy.ndarray`

    References
    ----------
    .. [1] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa.
       "Detection, location, and imaging of multiple scatterers by means
       of the iterative multiscaling method." IEEE transactions on
       microwave theory and techniques 52.4 (2004): 1217-1228.
    """
    binary = []

    # If a list of images
    if type(image) is list:
        for i in range(len(image)):
            hist, bins = np.histogram(image[i].flatten())
            T = bins[np.argsort(hist)[0]]
            binary.append(np.zeros(image[i].shape))
            binary[i][image[i] >= T] = 1.

    # Else a single image
    else:
        hist, bins = np.histogram(image.flatten())
        T = bins[np.argsort(hist)[0]]
        binary.append(np.zeros(image.shape))
        binary[image >= T] = 1.

    return binary


def filtering(image):
    """Noise filtering process.

    This process is intended to eliminate some artifacts of the image.
    This process is not well explained in [1]_ since the authors did not
    explained if a pixel is update considering the original or modified
    neighbors. Futhermore, a similar result as in Figure 2(d) in [1]_ is
    only obtained when only white pixels are considered. Therefore, we
    are implementing the rule only for white pixels.

    Parameters
    ----------
        image : :class:`numpy.ndarray` or a list of
            A list or a single image in matrix shape.

    References
    ----------
    .. [1] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa.
       "Detection, location, and imaging of multiple scatterers by means
       of the iterative multiscaling method." IEEE transactions on
       microwave theory and techniques 52.4 (2004): 1217-1228.
    """
    # List of images
    if type(image) is list:
        L = len(image)

        # For each image
        for k in range(L):
            NX, NY = image[k].shape

            # These matrices store the number of white and black
            # neighbors of each pixel.
            N0 = np.zeros((NY, NX), dtype=int)
            NMAX = np.zeros((NY, NX), dtype=int)

            # left-right, top-bottom
            for i in range(NX):
                for j in range(NY):

                    # Check every neighbor
                    m, n = neighbors(i, j, NX, NY)
                    for p in range(len(m)):
                        if image[k][n[p], m[p]] == 0:
                            N0[j, i] += 1
                        else:
                            NMAX[j, i] += 1

            # Change only white pixels with more black neighbors
            image[k][np.logical_and(image[k] == 0, N0 <= NMAX)] = 1

    # Single image
    else:

        # These matrices store the number of white and black neighbors
        # of each pixel.
        NX, NY = image.shape
        N0 = np.zeros((NY, NX), dtype=int)
        NMAX = np.zeros((NY, NX), dtype=int)

        # left-right, top-bottom
        for i in range(NX):
            for j in range(NY):

                # Check every neighbor
                m, n = neighbors(i, j, NX, NY)
                for p in range(len(m)):
                    if image[n[p], m[p]] == 0:
                        N0[j, i] += 1
                    else:
                        NMAX[j, i] += 1

        # Change only white pixels with more black neighbors
        image[np.logical_and(image == 0, N0 <= NMAX)] = 1


def object_detecting(binary):
    """Detect objects in the image.

    Detect objects in the image by a heuristic method. The method is
    described in [1]_. Each pixel is scanned and if it is black and no
    neighbor has a label, then a new label is assigned. Otherwise, the
    same label of the neighbors is assigned.

    Parameters
    ----------
        binary : :class:`numpy.ndarray` of a list of
            Binary (black-white) images in matrix shape. It may be a
            list or a single onel

    Returns
    -------
        labels : :class:`numpy.ndarray` of a list of
            Labeled image where 0's is nothing and 1, 2, ... and so on
            are objects.

        img : list
            In case of multiple images, each element in this list means
            which image the object belongs to.

    References
    ----------
    .. [1] Caorsi, Salvatore, Massimo Donelli, and Andrea Massa.
       "Detection, location, and imaging of multiple scatterers by means
       of the iterative multiscaling method." IEEE transactions on
       microwave theory and techniques 52.4 (2004): 1217-1228.
    """
    # Multiple areas
    if type(binary) is list:
        NIMG = len(binary)  # Number of images

        # Initialize a list in which each element is a image matrix
        labels = []
        for i in range(NIMG):
            labels.append(np.zeros(binary[i].shape, dtype=int))

        img = []  # which image the object belongs to
        NO = 0  # number of objects

        # For each image
        for ii in range(NIMG):
            NY, NX = binary[ii].shape

            # From left to right, top to bottom
            for i in range(NX):
                for j in range(NY):

                    # If the current pixel is black
                    if binary[ii][j, i] == 1:

                        # Look for labeled neighbors
                        m, n = neighbors(i, j, NX, NY)
                        for k in range(len(m)):

                            # If a neighbor has already been labeled
                            if labels[ii][n[k], m[k]] != 0:
                                labels[ii][j, i] = labels[ii][n[k], m[k]]
                                break

                            # New object is detected
                            elif k == len(m)-1:
                                NO += 1
                                labels[ii][j, i] = NO
                                img.append(ii)
                                # Give the same label to the neighbors
                                for p in range(len(m)):
                                    if (binary[ii][n[p], m[p]] == 1
                                            and labels[ii][n[p], m[p]] == 0):
                                        labels[ii][n[p], m[p]] = NO
        return labels, img

    # Single image
    else:
        NO = 0  # number of objects
        NY, NX = binary.shape

        # From left to right, top to bottom
        for i in range(NX):
            for j in range(NY):

                # If the current pixel is black
                if binary[j, i] == 1:

                    # Look for labeled neighbors
                    m, n = neighbors(i, j, NX, NY)
                    for k in range(len(m)):

                        # If a neighbor has already been labeled
                        if labels[n[k], m[k]] != 0:
                            labels[j, i] = labels[n[k], m[k]]
                            break

                        # New object is detected
                        elif k == len(m)-1:
                            NO += 1
                            labels[j, i] = NO
                            for p in range(len(m)):
                                if (binary[n[p], m[p]] == 1
                                        and labels[n[p], m[p]] == 0):
                                    labels[n[p], m[p]] = NO
        return labels


def new_areas(labels, binary, number_objects, x, y):
    """Determine new areas based on objects location.

    Given an image (or multiple) with the objects already detected, this
    process split the image into new ones with reduced areas, similar to
    a zooming process. The centroids of each object is computed as well
    as a side length for this new area which covers the object in a
    square fashion. This process is described in [1]_.

    Parameters
    ----------
        labels : :class:`numpy.ndarray` or a list of
            Set or a single image (matrix shape) with the labels of the
            objects in each pixel.

        binary : :class:`numpy.ndarray` or list of
            Binarized images.

        number_objects : int
            Number of objects detected in the image

        x, y : :class:`numpy.ndarray` or list of
            Cartesian coordinates of each image.

    Returns
    -------
        xc, yc, L : :class:`numpy.ndarray`
            Centroids and length of each area.

    References
    ----------
    .. [1] Caorsi, Salvatore, et al. "A new methodology based on an
       iterative multiscaling for microwave imaging." IEEE transactions
       on microwave theory and techniques 51.4 (2003): 1162-1173.
    """
    NO = number_objects
    xc, yc, L = np.zeros((2, NO)), np.zeros((2, NO)), np.zeros((2, NO))

    # Multiple images
    if type(labels) is list:
        NIMG = len(labels)

        # Compute numerator and denominator terms of the expression
        for k in range(NIMG):
            NY, NX = binary[k].shape
            for i in range(NX):
                for j in range(NY):
                    if labels[k][j, i] != 0:
                        xc[0, labels[k][j, i]-1] += x[k][j, i]*binary[k][j, i]
                        xc[1, labels[k][j, i]-1] += binary[k][j, i]
                        yc[0, labels[k][j, i]-1] += y[k][j, i]*binary[k][j, i]
                        yc[1, labels[k][j, i]-1] += binary[k][j, i]

    # Single image
    else:

        # Compute numerator and denominator terms of the expression
        NY, NX = binary.shape
        for i in range(NX):
            for j in range(NY):
                if labels[j, i] != 0:
                    xc[0, labels[j, i]-1] += x[j, i]*binary[j, i]
                    xc[1, labels[j, i]-1] += binary[j, i]
                    yc[0, labels[j, i]-1] += y[j, i]*binary[j, i]
                    yc[1, labels[j, i]-1] += binary[j, i]

    # Equation (2) in [1]
    xc = xc[0, :]/xc[1, :]
    yc = yc[0, :]/yc[1, :]

    # Multiple images
    if type(labels) is list:

        # Compute lengths
        RHO_MAX = np.zeros(NO)
        for k in range(len(labels)):
            NY, NX = binary[k].shape
            for i in range(NX):
                for j in range(NY):
                    if labels[k][j, i] != 0:
                        lab = labels[k][j, i]
                        rho = np.sqrt((xc[lab-1]-x[k][j, i])**2
                                      + (yc[lab-1]-y[k][j, i])**2)
                        if rho > RHO_MAX[lab-1]:
                            RHO_MAX[lab-1] = rho
                        L[0, lab-1] += rho*binary[k][j, i]
                        L[1, lab-1] += binary[k][j, i]

    # Single image
    else:
        NY, NX = binary.shape
        for i in range(NX):
            for j in range(NY):
                if labels[j, i] != 0:
                    lab = labels[j, i]
                    rho = np.sqrt((xc[lab-1]-x[j, i])**2
                                  + (yc[lab-1]-y[j, i])**2)
                    if rho > RHO_MAX[lab-1]:
                        RHO_MAX[lab-1] = rho
                    L[0, lab-1] += rho*binary[j, i]
                    L[1, lab-1] += binary[j, i]

    # Equation (3) in [1]
    L = 2*L[0, :]/L[1, :]
    return xc, yc, L
