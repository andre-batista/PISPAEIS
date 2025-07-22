import time as tm
from scipy.optimize import minimize, OptimizeResult, differential_evolution
import analytical as ana
import stochastic as stc
import sys
import inputdata as ipt
import result as rst
import configuration as cfg
import pickle
from joblib import Parallel, delayed
import multiprocessing

POSITION_RANGE = 'position_range'
RADIUS_RANGE = 'radius_range'
CONTRAST_RANGE = 'contrast_range'

class CircleApproximation(stc.Stochastic):
    def __init__(self, outputmode, number_executions=1, position_range=None,
                 radius_range=None, contrast_range=None, alias='ca',
                 parallelization=False, solver="de"):
        super().__init__(outputmode, alias=alias,
                         parallelization=parallelization,
                         number_executions=number_executions)
        if position_range is not None:
            self.position_range = [position_range[0], position_range[1], 
                                   position_range[2], position_range[3]]
        else:
            self.position_range = None
        if radius_range is not None:
            if isinstance(radius_range, (int, float)):
                self.radius_range = [0.001, radius_range]
            elif isinstance(radius_range, (list, tuple)):
                self.radius_range = [radius_range[0], radius_range[1]]
        else:
            self.radius_range = None
        if contrast_range is not None:
            if isinstance(contrast_range, (int, float)):
                self.contrast_range = [0., contrast_range]
            elif isinstance(contrast_range, (list, tuple)):
                self.contrast_range = [contrast_range[0], contrast_range[1]]
        else:
            self.contrast_range = [0., 10.]
        self.solver = solver
            
    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        result = super().solve(inputdata, discretization, print_info, 
                               print_file)

        if self.position_range is not None:
            position_range = self.position_range
        else:
            xmin, xmax = cfg.get_bounds(inputdata.configuration.Lx)
            ymin, ymax = cfg.get_bounds(inputdata.configuration.Ly)
            position_range = [xmin, xmax, ymin, ymax]
        if self.radius_range is not None:
            radius_range = self.radius_range
        else:
            radius_range = [0.001, min([inputdata.configuration.Lx,
                                     inputdata.configuration.Ly])/2]
        contrast_range = self.contrast_range
        
        bounds = [(position_range[0], position_range[1]),
                  (position_range[2], position_range[3]),
                  (radius_range[0], radius_range[1]),
                  (contrast_range[0], contrast_range[1])]
        
        initial_guess = [(position_range[0] + position_range[1])/2,
                         (position_range[2] + position_range[3])/2,
                         radius_range[0] + (radius_range[1]
                                            - radius_range[0])*.5,
                         contrast_range[0] + (contrast_range[1]
                                              - contrast_range[0])*.1]
        
        if contrast_range[1] > 1.:
            initial_guess[3] = 1.
        
        if print_info and not self.parallelization:
            callback = mycallback
            global iteration, execution
            iteration = 0
            execution = 0
        else:
            callback = None
            
        run_names = [result.name + '_exec%d' % ne for ne in range(self.nexec)]
        
        if self.parallelization:
            if print_info:
                print('Running executions in parallel...', file=print_file)

            num_cores = multiprocessing.cpu_count()
            output = (Parallel(n_jobs=num_cores))(delayed(self._run_algorithm)
                                                  (bounds, inputdata, callback,
                                                   run_names[ne], initial_guess)
                                                  for ne in range(self.nexec))
        else:
            output = []
            for ne in range(self.nexec):
                output.append(self._run_algorithm(bounds, inputdata, callback,
                                                   run_names[ne],
                                                   initial_guess))
                if print_info:
                    iteration = 0
                    execution += 1

        result = self.outputmode.make(inputdata.name + '_' + self.alias,
                                      self.alias, output)

        return result

    def _run_algorithm(self, bounds, inputdata, callback, run_name,
                       initial_guess=None):
        result = rst.Result(run_name,
                            method_name=self.alias,
                            configuration=inputdata.configuration)

        tic = tm.time()

        if self.solver == "de":
            solution = differential_evolution(
                objfun, bounds, args=(inputdata,), strategy='best1bin', maxiter=50,
                popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
                seed=None, callback=callback, disp=False, polish=True,
                init='latinhypercube', atol=0, updating='deferred', workers=-1, 
                constraints=(), x0=None, integrality=None, vectorized=False
            )
        else:

            solution = minimize(objfun, initial_guess, args=(inputdata,),
                                bounds=bounds, method='L-BFGS-B',
                                callback=callback, options={'ftol':1e-8, 
                                                            'disp':False})

        execution_time = tm.time() - tic

        circle = build_solution(solution.x, inputdata.configuration,
                                SAVE_INTERN_FIELD=True, SAVE_MAP=True,
                                resolution=inputdata.resolution)
        
        contrast = cfg.get_contrast_map(circle.rel_permittivity,
                                        circle.conductivity,
                                        configuration=inputdata.configuration)
        
        result.update_error(inputdata, scattered_field=circle.scattered_field,
                            total_field=circle.total_field, contrast=contrast,
                            objective_function=solution.fun)

        result.scattered_field = circle.scattered_field
        result.total_field = circle.total_field
        result.rel_permittivity = circle.rel_permittivity
        result.conductivity = circle.conductivity
        result.path = solution.x
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        if rst.NUMBER_ITERATIONS in inputdata.indicators:
            result.number_iterations = iteration
        if rst.NUMBER_EVALUATIONS in inputdata.indicators:
            result.number_evaluations = solution.nfev
        return result

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print('Position range:', self.position_range, file=print_file)
        print('Radius range:', self.radius_range, file=print_file)
        print('Contrast range:', self.contrast_range, file=print_file)

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[POSITION_RANGE] = self.position_range
        data[RADIUS_RANGE] = self.radius_range
        data[CONTRAST_RANGE] = self.contrast_range
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.position_range = data[POSITION_RANGE]
        self.radius_range = data[RADIUS_RANGE]
        self.contrast_range= data[CONTRAST_RANGE]

    def copy(self, new=None):
        if new is None:
            return CircleApproximation(self.outputmode,
                                       number_executions=self.nexec,
                                       position_range=self.position_range,
                                       radius_range=self.radius_range,
                                       contrast_range=self.contrast_range,
                                       alias=self.alias,
                                       parallelization=self.parallelization)
        else:
            super().copy(new)
            self.outputmode = new.outputmode
            self.nexec = new.nexec
            self.position_range = new.position_range
            self.radius_range = new.radius_range
            self.contrast_range = new.contrast_range

    def __str__(self):
        message = super().__str__()
        message += 'Position range: ' + str(self.position_range) + '\n'
        message += 'Radius range: ' + str(self.radius_range) + '\n'
        message += 'Contrast range: ' + str(self.contrast_range)
        return message

    
def build_solution(x, configuration, SAVE_INTERN_FIELD=False,
                   SAVE_MAP=False, resolution=(20, 20)):
    position = x[:2]
    radius = x[2]
    contrast = x[3]
    
    solver = ana.Analytical(contrast=contrast, position=position,
                            radius=radius)
    
    test = ipt.InputData(name='test', configuration=configuration,
                         resolution=resolution)
    
    if configuration.good_conductor:
        solver.conductor_cylinder(test, SAVE_INTERN_FIELD=SAVE_INTERN_FIELD, 
                                  SAVE_MAP=SAVE_MAP)
    else:
        solver.dielectric_cylinder(test, SAVE_INTERN_FIELD=SAVE_INTERN_FIELD, 
                                   SAVE_MAP=SAVE_MAP)
    
    return test


def objfun(x, data):
    test = build_solution(x, data.configuration)
    return rst.compute_rre(data.scattered_field, test.scattered_field)


def mycallback(intermediate_result: OptimizeResult):
    global iteration
    if iteration == 0:
        print(f'--------------------------------------------------')
        print(f'Execution: {execution}')
    iteration += 1
    print(f'Iteration: {iteration} - f(x): {intermediate_result.fun}')
    
    