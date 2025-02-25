import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

class SystemParabolicFEM(object):

    def __init__(self, mesh, time_interval, temp_points=11, num_equations=2):
        self.mesh = mesh
        self.n_mesh_p = mesh.coordinates().shape[0]
        self.num_eq = num_equations
        self.time_interval = time_interval
        self.temp_points = temp_points
        self.V = self._initialize_problem()
        self.inner_product = self._inner_product()
        self._sol = None
        self._real_sol = None
        self._mu = None
    
    def _initialize_problem(self):
        return fe.VectorFunctionSpace(self.mesh, 'P', 1, dim=self.num_eq)
    
    def _inner_product(self):
        u = fe.TestFunction(self.V)
        v = fe.TrialFunction(self.V)
        return fe.assemble(fe.inner(fe.grad(u[0]), fe.grad(v[0])) * fe.dx + fe.inner(fe.grad(u[1]), fe.grad(v[1])) * fe.dx)
    
    def _initialize_fem_problem(self, mu, t_0, T):
        if abs(t_0-0) < np.finfo(np.float64).eps:
            h = T / (self.temp_points-1)
            # Relativo alle condizioni iniziali
            u_n = fe.Function(self.V)
            u_n.interpolate(self._initial_condition(mu))
            initial = np.array(u_n.vector().get_local())
            # Memorizzare soluzione
            lst_solution_matrix = []
            for i in range(self.num_eq):
                lst_solution_matrix.append(np.zeros((self.n_mesh_p, self.temp_points)))
                lst_solution_matrix[i][:, 0] = initial[i*self.n_mesh_p:(i+1)*self.n_mesh_p]
            return h, u_n, lst_solution_matrix
        else:
            raise NotImplementedError("Only t_0 = 0")
    
    def _boundary_cond(self, mu, t_point):
        u_D = fe.Expression(
            (
                "exp(var1*t) + var2*x[0] + var3*x[1] + x[2]",
                "exp(var1*t) + var2*x[0]*x[0] + var3*x[1]*x[1] + x[2]*x[2]"
            ),
            degree=3,
            var1=mu[0], var2=mu[1], var3=mu[2], t=t_point,
        )
        return u_D
    
    def _initial_condition(self, mu):
        u_0 = fe.Expression(
            (
                "1 + var2*x[0] + var3*x[1] + x[2]",
                "1 + var2*x[0]*x[0] + var3*x[1]*x[1] + x[2]*x[2]"
            ),
            degree=3,
            var2=mu[1], var3=mu[2]
        )
        return u_0
    
    def _problem_forms(self, mu, u_n, t_point, h):
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        # Definizione della forma bilineare e lineare per l'equazione del calore
        a = u[0]*v[0]*fe.dx + h*fe.dot(fe.grad(u[0]), fe.grad(v[0]))*fe.dx + u[1]*v[1]*fe.dx + h*fe.dot(fe.grad(u[1]), fe.grad(v[1]))*fe.dx

        exp_f = fe.Expression(
            (
                "var1*exp(var1*t)",
                "var1*exp(var1*t) - 2 * (var2 + var3 + 1)"
            ),
            degree=3,
            var1 = mu[0], var2 = mu[1], var3=mu[2], t=t_point
        )
        L = (u_n[0] + h * exp_f[0]) * v[0] * fe.dx + (u_n[1] + h * exp_f[1]) * v[1] * fe.dx
        return a, L
    
    def _single_step(self, mu, step, h, u_n):
        t_point = h*step
        u_D = self._boundary_cond(mu, t_point)
        bc = fe.DirichletBC(self.V, u_D, fe.DomainBoundary())
        bcs = []
        bcs.append(bc)
        a, L = self._problem_forms(mu, u_n, t_point, h)
        A, b = fe.assemble_system(a, L, bcs)
        solution = fe.Function(self.V)
        fe.solve(A, solution.vector(), b)
        return solution

    def solve_fem(self, mu):
        t_0, T = self.time_interval
        h, u_n, lst_solution_matrix = self._initialize_fem_problem(mu, t_0, T)
        lst_solutions = []
        for step in range(1, self.temp_points):
            sol = self._single_step(mu, step, h, u_n)
            num_sol = np.array(sol.vector().get_local())
            for i in range(self.num_eq):
                lst_solution_matrix[i][:, step] = num_sol[i*self.n_mesh_p:(i+1)*self.n_mesh_p]
            u_n.assign(sol)
            lst_solutions.append(sol)
        return lst_solution_matrix, lst_solutions
    
    def _exact_exp(self, mu, t_point):
        exact_exp = fe.Expression(
            (
                "exp(var1*t) + var2*x[0] + var3*x[1] + x[2]",
                "exp(var1*t) + var2*x[0]*x[0] + var3*x[1]*x[1] + x[2]*x[2]"
            ),
            degree=3,
            var1=mu[0], var2=mu[1], var3=mu[2], t=t_point,
        )
        return exact_exp
    
    def exact_solution(self, mu):
        h = 1 / (self.temp_points - 1)
        lst_functions = []
        lst_exact_matrix = []
        for i in range(self.num_eq):
            lst_exact_matrix.append(np.zeros((self.mesh.coordinates().shape[0], self.temp_points)))
        u = fe.Function(self.V)
        lst_functions.append(u)
        for step in range(self.temp_points):
            t_point = h*step
            u.interpolate(self._exact_exp(mu, t_point))
            lst_functions.append(u)
            u_num = np.array(u.vector().get_local())
            for i in range(self.num_eq):
                lst_exact_matrix[i][:, step] = u_num[i*self.n_mesh_p:(i+1)*self.n_mesh_p]
        self._real_sol = lst_exact_matrix
        return lst_exact_matrix, lst_functions

    @staticmethod
    def _compute_error_matrix(approx, exact, text):
        err_a = np.max(np.linalg.norm(approx - exact, axis=1))
        err_r = err_a / np.max(np.linalg.norm(exact, axis=1))
        if text:
            print(f"Errore Assoluto: {err_a:.4e}")
            print(f"Errore Relativo: {err_r:.4e}")
        return err_a, err_r
    
    def compute_error(self, lst_approx, lst_exact, text=True):
        lst_err_a, lst_err_r = [], []
        for i in range(self.num_eq):
            if text:
                print(f"ERRORI EQUAZIONE {i+1}")
            err_a, err_r = self._compute_error_matrix(lst_approx[i], lst_exact[i], text)
            lst_err_a.append(err_a)
            lst_err_r.append(err_r)
        return lst_err_a, lst_err_r
    
    @property
    def sol(self):
        if self._sol is None:
            raise AttributeError("No solution found. Call the method solve_fem.")
        return self._sol
    
    @property
    def real_sol(self):
        if self._real_sol is None:
            raise AttributeError("No solution found. Call the method exact_solution.")
        return self._real_sol
    
    @property
    def mu(self):
        if self._mu is None:
            raise AttributeError("No parameter found. Call the method solve_fem.")
        return self._mu