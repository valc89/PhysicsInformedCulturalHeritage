import numpy as np
import pandas as pd
import fenics as fe
import rbnics as rb
import matplotlib.pyplot as plt

from time import time
from typing import List
from statistics import mean
from tabulate import tabulate
from collections import defaultdict
from fem_problems.invTP2.finite_element import SystemParabolicFEM

class TemporalPOD(object):

    def __init__(self, mu_range : List[tuple], fem_p : SystemParabolicFEM):
        self.mu_range = mu_range
        self.fem_p = fem_p
        self.total_parameters = len(mu_range)
        self.pod = rb.backends.dolfin.ProperOrthogonalDecomposition(self.fem_p.V, self.fem_p.inner_product)
        self._Z = None
        self._eigenvalues, self._basis_functions, self._num_basis = None, None, None
        self._dict_pods = None
        self._training_set = None
        self._testing_set = None
        self._Nmax, self._tol = None, None
        self._data_solution = None
    
    def sampling_parameters(self, tot_samples : int, test = False):
        training =  rb.sampling.distributions.UniformDistribution().sample(self.mu_range, tot_samples)
        if not test:
            self._training_set = np.array(training)
        else:
            self._testing_set = np.array(training)
        return training
    
    def plot_samples(self, figsize = (12, 6), save=True, filename="samples_plot.png"):
        if not self._training_set is None:
            if not self._testing_set is None:
                if self.total_parameters != 1:
                    self._plot_both(figsize, save, filename)
                else:
                    self._plot_singlepar_trtst(figsize, save, filename)
            else:
                if self.total_parameters != 1:
                    self._plot_single(figsize, save, filename)
                else:
                    self._single_plot(figsize, save, filename)
        else:
            raise AttributeError("No training set found. Call the method sampling_parameters")
    
    def check_orthogonality_temporal_pod(self, i, j, pos):
        Z_i = self.dict_pods[pos]["basis_function"][i]
        Z_j = self.dict_pods[pos]["basis_function"][j]
        inn_pr = self.fem_p.inner_product
        delta_ij = rb.backends.dolfin.transpose(Z_i)*inn_pr*Z_j
        print(f"Check orthogonality for basis functions {pos} i = {i} and j = {j}: {delta_ij}")
    
    def pod_execution(self, N_max, tol):
        self._dict_pods = self._initialize_single_pods(self.training_set.shape[0])
        self._store_samples()
        N_eval = self._temporal_pod_compressions(self.fem_p.temp_points, tol*1e-5)
        if N_max > N_eval:
            N_max = N_eval
        self._Nmax, self._tol = N_max, tol
        self._parameter_storing()
        self._parameter_pod_compression(N_max, tol)
    
    def check_orthogonality(self, i, j):
        delta_ij = rb.backends.dolfin.transpose(self.basis_functions[i])*self.fem_p.inner_product*self.basis_functions[j]
        print(f"Check orthogonality for basis functions i = {i} and j = {j}: {delta_ij}")
    
    def store_reduction(self, directory, filename):
        print("Start STORING REDUCTION...", end="")
        self.Z.save(directory=directory, filename=filename)
        print("...End STORING REDUCTION")
    
    def load_reduction(self, directory, filename):
        print("Start LOADING REDUCTION...", end="")
        self._Z = rb.backends.dolfin.BasisFunctionsMatrix(self.fem_p.V)
        self._Z.init("u")
        self._Z.load(directory=directory, filename=filename)
        self._eigenvalues, self._basis_functions, self._num_basis = None, self._Z, np.array(self._Z).shape[0]
        self._training_set = None
        self._testing_set = None
        print("...End LOADING REDUCTION")
    
    def plot_each_eigenvalues(self, fig_size = (6, 18)):
        tot_plots = len(self.dict_pods)
        _, ax = plt.subplots(nrows=tot_plots, ncols=1, figsize=fig_size)
        for i in range(tot_plots):
            ax[i].plot(np.arange(1, self.dict_pods[i]["N"]+1), np.array(self.dict_pods[i]["eigenvalues"]), "*-")
            ax[i].set_title(f"Eigenvalues parameter mu = {self.training_set[i]}")
            ax[i].set_xlabel("order")
            ax[i].set_ylabel("$\lambda$")
            ax[i].set_yscale("log")
        plt.tight_layout()
        plt.show()
    
    def plot_eigenvalues(self, save=True, filename="plot_eig.png"):
        y = np.array(self.eigenvalues)
        n = self.num_basis
        x = np.arange(1, n+1)
        plt.plot(x, y, "*-")
        plt.title("Eigenvalues")
        plt.xlabel("order")
        plt.ylabel("$\lambda$")
        plt.yscale("log")
        plt.tight_layout()
        if save:
            plt.savefig(filename, transparent=True)
        plt.show()
    
    def solve_rom(self, mu, N):
        t_0, T = self.fem_p.time_interval
        h, u_n, lst_solution_matrix = self.fem_p._initialize_fem_problem(mu, t_0, T)
        lst_solutions = []
        for step in range(1, self.fem_p.temp_points):
            sol = self._single_step(mu, step, h, u_n, N)
            sol_num = np.array(sol.vector().get_local())
            for i in range(self.fem_p.num_eq):
                lst_solution_matrix[i][:, step] = sol_num[i*self.fem_p.n_mesh_p:(i+1)*self.fem_p.n_mesh_p]
            u_n.assign(sol)
            lst_solutions.append(sol)
        return lst_solution_matrix, lst_solutions
    
    def solve_online_rom(self, mu, N):
        t_0, T = self.fem_p.time_interval
        h, u_n, lst_solution_matrix = self.fem_p._initialize_fem_problem(mu, t_0, T)
        lst_solutions = []
        for step in range(1, self.fem_p.temp_points):
            sol = self._single_step(mu, step, h, u_n, N)
            sol_num = np.array(sol.compute_vertex_values(self.fem_p.mesh))
            for i in range(self.fem_p.num_eq):
                lst_solution_matrix[i][:, step] = sol_num[i*self.fem_p.n_mesh_p:(i+1)*self.fem_p.n_mesh_p]
            u_n.assign(sol)
            lst_solutions.append(sol)
        return lst_solution_matrix, lst_solutions
    
    def error_analysis(self):
        error = defaultdict(dict)
        output_error = defaultdict(dict)
        for N_basis in range(1, self.num_basis+1):
            print(f"\n=============== POD basis number N = {N_basis} ===============")
            error["abs"][N_basis] = []
            error["rel"][N_basis] = []
            error["speed_up"][N_basis] = []
            output_error["abs"][N_basis] = []
            output_error["rel"][N_basis] = []
            for j in range(self.testing_set.shape[0]):
                mu = self.testing_set[j]
                print(f'Solve for test parameter i = {j}, mu={mu}')
                t0 = time()
                u_fom, _ = self.fem_p.solve_fem(mu)
                t_fom = time()
                u_rom, _ = self.solve_rom(mu, N_basis)
                t_rom = time()
                assl, rel = self.fem_p.compute_error(u_fom, u_rom, text=False)
                speed_up = (t_fom-t0)/(t_rom-t_fom)
                error["abs"][N_basis].append(mean(assl))
                error["rel"][N_basis].append(mean(rel))
                error["speed_up"][N_basis].append(speed_up)
        return error, output_error
    
    def plot_errors(self, error, save=True, filename="plot_errors.png"):
        basis_range = np.array(range(1, self.num_basis+1))
        plt.plot(basis_range, np.array([mean(error["abs"][i]) for i in basis_range]), '-o', label = 'abs_mean_errors')
        plt.plot(basis_range, np.array([max(error["abs"][i]) for i in basis_range]), '-.o', label = 'abs_max_errors')
        plt.plot(basis_range, np.array([mean(error["rel"][i]) for i in basis_range]), '-s', label = 'rel_mean_errors')
        plt.plot(basis_range, np.array([max(error["rel"][i]) for i in basis_range]), '-.s', label = 'rel_max_errors')
        plt.xlabel("N")
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()
    
    def plot_speed_up(self, error, save=True, filename="plt_speed_up.png"):
        basis_range = np.array(range(1, self.num_basis+1))
        plt.plot(basis_range, np.array([mean(error["speed_up"][i]) for i in basis_range]), '-o', label = 'mean_speed_up')
        plt.plot(basis_range, np.array([max(error["speed_up"][i]) for i in basis_range]), '-.s', label = 'max_speed_up')
        plt.plot(basis_range, np.array([min(error["speed_up"][i]) for i in basis_range]), '-.^', label = 'min_speed_up')
        plt.xlabel("N")
        plt.grid()
        plt.legend()
        if save:
            plt.savefig(filename, transparent=True)
        plt.show()
    
    def table_error(self, error):
        if self._data_solution is None:
            self._create_data_solution(error)
        headers = ["N", "mean(error_u)", "max(error_u)", "mean(relative_error_u)", "max(relative_error_u)", "speed_up"]
        print("Solution errors:")
        print(tabulate(self.data_solution, headers=headers, tablefmt="grid"))
    
    def save_table_error(self, error, path="", filename="table_error"):
        if self._data_solution is None:
            self._create_data_solution(error)
        df = pd.DataFrame(
            data=self.data_solution,
            columns=["N", "mean(error_u)", "max(error_u)", "mean(relative_error_u)", "max(relative_error_u)", "speed_up"],
        )
        df.to_csv(path+filename+".csv", sep=";", index=False)
    
    def _create_data_solution(self, error):
        self._data_solution = []
        basis_range = np.array(range(1, self.num_basis+1))
        abs_errors_mean = np.array([mean(error["abs"][i]) for i in basis_range])
        abs_errors_max = np.array([max(error["abs"][i]) for i in basis_range])
        rel_errors_mean = np.array([mean(error["rel"][i]) for i in basis_range])
        rel_errors_max = np.array([max(error["rel"][i]) for i in basis_range])
        speed_up_mean = np.array([mean(error["speed_up"][i]) for i in basis_range])
        for i in range(len(abs_errors_mean)):
            self._data_solution.append((i+1, abs_errors_mean[i], abs_errors_max[i], rel_errors_mean[i], rel_errors_max[i], speed_up_mean[i]))
    
    def _single_step(self, mu, step, h, u_n, N):
        t_point = h*step
        u_D = self.fem_p._boundary_cond(mu, t_point)
        bc = fe.DirichletBC(self.fem_p.V, u_D, fe.DomainBoundary())
        bcs = []
        bcs.append(bc)
        a, L = self.fem_p._problem_forms(mu, u_n, t_point, h)
        A, b = fe.assemble_system(a, L, bcs)
        reduced_A = rb.backends.dolfin.transpose(self.Z[:N])*A*self.Z[:N]
        reduced_b = rb.backends.dolfin.transpose(self.Z[:N])*b
        reduced_solution = rb.backends.online.numpy.Function(N)
        reduced_solver = rb.backends.online.numpy.LinearSolver(
            reduced_A,
            reduced_solution,
            reduced_b
        )
        reduced_solver.solve()
        return self.Z[:N]*reduced_solution
    
    def _initialize_single_pods(self, num):
        lst = []
        for i in range(num):
            dz = {
                "pod" : rb.backends.dolfin.ProperOrthogonalDecomposition(self.fem_p.V, self.fem_p.inner_product),
                "eigenvalues" : [],
                "basis_function" : None,
                "N" : None,
                "Z" : None
            }
            lst.append(dz)
        return lst.copy()
    
    def _parameter_storing(self):
        for element in self.dict_pods:
            count = 0
            for basis in element["basis_function"]:
                if count < element["N"]:
                    self.pod.store_snapshot(basis)
                count += 1
    
    def _parameter_pod_compression(self, N_max, tol):
        print("PARAMETER POD COMPRESSION EXECUTION...")
        self.pod.eigenvalues = []
        self.pod.retained_energy = []
        eigenvalues, _, basis_functions, N = self.pod.apply(N_max, tol)
        self._eigenvalues, self._basis_functions, self._num_basis = eigenvalues, basis_functions, N
        self._Z = rb.backends.dolfin.BasisFunctionsMatrix(self.fem_p.V)
        self._Z.init('u')
        self._Z.enrich(basis_functions)
        print(f"Number of selected basis: {N}")
        print("...End of PARAMETER POD COMPRESSION EXECUTION")
        
        
    def _temporal_pod_compressions(self, N_max, tol):
        tot_N = 0
        for element in self.dict_pods:
            print("TEMPORAL STEPS POD COMPRESSION EXECUTION...")
            element = self._single_temporal_compression(element, N_max, tol)
            tot_N += element["N"]
            print("...End of TEMPORAL STEPS POD COMPRESSION EXECUTION")
        return tot_N
    
    def _single_temporal_compression(self, element, N_max, tol):
        element["eigenvalues"].clear()
        element["pod"].eigenvalues = []
        element["pod"].retained_energy = []
        element["eigenvalues"], _, element["basis_function"], element["N"] = element["pod"].apply(N_max, tol)
        Z = rb.backends.dolfin.BasisFunctionsMatrix(self.fem_p.V)
        Z.init('u')
        Z.enrich(element["basis_function"])
        element["Z"] = Z
        return element

    def _store_samples(self):
        print("STORING SNAPSHOTS...")
        for i in range(self.training_set.shape[0]):
            print(f'+++ Compute and store temporal snaphot {i+1} for mu = {self.training_set[i]} +++')
            self._store_temporal_snapshot(i)
            print(f"... End of compute and store temporal snaphot {i+1}")
        print("... End of STORING SNAPSHOTS")
    
    def _store_temporal_snapshot(self, i):
        print(f"STORING TEMPORAL SNAPSHOT")
        _, lst_sol = self.fem_p.solve_fem(self.training_set[i])
        for element in lst_sol:
            self.dict_pods[i]["pod"].store_snapshot(element)
        
    
    def _plot_both(self, fig_size = (12, 12), save=True, filename="samples_plot.png"):
        n = self.total_parameters
        x = np.arange(self.training_set.shape[0])
        x_t = np.arange(self.testing_set.shape[0])
        _, ax = plt.subplots(nrows=n, ncols=2, figsize=fig_size)
        for count in range(n):
            ax[count, 0].plot(x, self.training_set[:, count], "^")
            ax[count, 0].set_title(f"Training set parameter $\mu_{count+1}$")
            ax[count, 0].set_xlabel("samples")
            ax[count, 0].set_ylabel(f"$\mu_{count+1}$")
            ax[count, 0].set_ylim(self.mu_range[count])

            ax[count, 1].plot(x_t, self.testing_set[:, count], "^r")
            ax[count, 1].set_title(f"Testing set parameter $\mu_{count+1}$")
            ax[count, 1].set_xlabel("samples")
            ax[count, 1].set_ylabel(f"$\mu_{count+1}$")
            ax[count, 1].set_ylim(self.mu_range[count])
        plt.tight_layout()
        if save:
            plt.savefig(filename, transparent=True)
        plt.show()

    
    def _plot_single(self, fig_size = (6, 15), save=True, filename="samples_plot.png"):
        n = self.total_parameters
        x = np.arange(self.training_set.shape[0])
        _, ax = plt.subplots(nrows=n, ncols=1, figsize=fig_size)
        for count in range(n):
            ax[count].plot(x, self.training_set[:, count], "^")
            ax[count].set_title(f"Training set parameter $\mu_{count+1}$")
            ax[count].set_xlabel("samples")
            ax[count].set_ylabel(f"$\mu_{count+1}$")
            ax[count].set_ylim(self.mu_range[count])
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, transparent=True)
        plt.show()
    
    def _plot_singlepar_trtst(self, fig_size = (11, 6), save=True, filename="samples_plot.png"):
        x = np.arange(self.training_set.shape[0])
        x_t = np.arange(self.testing_set.shape[0])
        count = 0
        _, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
        ax[0].plot(x, self.training_set[:, count], "^")
        ax[0].set_title(f"Training set parameter $\mu_{count+1}$")
        ax[0].set_xlabel("samples")
        ax[0].set_ylabel(f"$\mu_{count+1}$")
        ax[0].set_ylim(self.mu_range[count])

        ax[1].plot(x_t, self.testing_set[:, count], "^r")
        ax[1].set_title(f"Testing set parameter $\mu_{count+1}$")
        ax[1].set_xlabel("samples")
        ax[1].set_ylabel(f"$\mu_{count+1}$")
        ax[1].set_ylim(self.mu_range[count])
        
        plt.tight_layout()
        if save:
            plt.savefig(filename, transparent=True)
        plt.show()
    
    def _single_plot(self, fig_size = (7, 5), save=True, filename="samples_plot.png"):
        count = 0
        x = np.arange(self.training_set.shape[0])
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        ax.plot(x, self.training_set[:, count], "^")
        ax.set_title(f"Training set parameter $\mu_{count+1}$")
        ax.set_xlabel("samples")
        ax.set_ylabel(f"$\mu_{count+1}$")
        ax.set_ylim(self.mu_range[count])
        plt.tight_layout()
        if save:
            plt.savefig(filename, transparent=True)
        plt.show()
    
    @property
    def dict_pods(self):
        if self._dict_pods is None:
            raise AttributeError("No dict_pods found. Call the method pod_execution")
        return self._dict_pods

    @property
    def training_set(self):
        if self._training_set is None:
            raise AttributeError("No training set found. Call the method sampling_parameters")
        return self._training_set
    
    @property
    def testing_set(self):
        if self._testing_set is None:
            raise AttributeError("No training set found. Call the method sampling_parameters")
        return self._testing_set
    
    @property
    def Nmax(self):
        if self._Nmax is None:
            raise AttributeError("No Nmax found. Call the method pod_execution")
        return self._Nmax
    
    @property
    def tol(self):
        if self._tol is None:
            raise AttributeError("No tol found. Call the method pod_execution")
        return self._tol
    
    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            raise AttributeError("No eigenvalues found. Call the method pod_execution")
        return self._eigenvalues
    
    @property
    def basis_functions(self):
        if self._basis_functions is None:
            raise AttributeError("No basis_functions found. Call the method pod_execution")
        return self._basis_functions
    
    @property
    def num_basis(self):
        if self._num_basis is None:
            raise AttributeError("No num_basis found. Call the method pod_execution")
        return self._num_basis
    
    @property
    def Z(self):
        if self._Z is None:
            raise AttributeError("No num_basis found. Call the method pod_execution")
        return self._Z
    
    @property
    def data_solution(self):
        if self._data_solution is None:
            raise AttributeError("No data_solution found. Call the method table_error or save_table_error")
        return self._data_solution