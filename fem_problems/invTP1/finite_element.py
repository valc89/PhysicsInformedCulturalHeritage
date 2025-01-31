import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

class PoissonFEM(object):

    def __init__(self, mesh):
        self.mesh = mesh
        self.n_mesh_p = mesh.coordinates().shape[0]
        self.V = self._initialize_problem()
        self.inner_product = self._inner_product()
        self._sol = None
        self._real_sol = None
        self._mu = None
        self.pi = fe.Constant(np.pi)
    
    def _initialize_problem(self):
        V_el = fe.FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
        V = fe.FunctionSpace(self.mesh, V_el)
        return V
    
    def _inner_product(self):
        u = fe.TestFunction(self.V)
        v = fe.TrialFunction(self.V)
        return fe.assemble(fe.inner(fe.grad(u), fe.grad(v))*fe.dx)
    
    def _boundary(self, mu):
        u_D = fe.Expression(
            "var1 * x[0] * cos(var2 * pi * x[1]) * sin(var3 * pi * x[2])",
            degree=3,
            var1=mu[0], var2=mu[1], var3=mu[2], pi=self.pi,
        )
        bc = fe.DirichletBC(self.V, u_D, fe.DomainBoundary())
        bcs = []
        bcs.append(bc)
        return bcs
    
    def _problem_forms(self, mu):
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        a = - fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        exp_f = fe.Expression(
            "- (var2*var2 + var3*var3) * pi * pi * var1 * x[0] * cos(var2*pi*x[1])*sin(var3*pi*x[2])",
            degree=3,
            var1=mu[0], var2=mu[1], var3=mu[2], pi=self.pi
        )
        f = exp_f * v * fe.dx
        return a, f

    def solve_fem(self, mu):
        self._mu = mu
        a, f = self._problem_forms(mu)
        bcs = self._boundary(mu)
        A, b = fe.assemble_system(a, f, bcs)
        solution = fe.Function(self.V)
        fe.solve(A, solution.vector(), b)
        num_sol = np.array(solution.vector().get_local())
        return num_sol, solution
    
    def _exact_exp(self, mu):
        exact_exp = fe.Expression(
            "var1 * x[0] * cos(var2 * pi * x[1]) * sin(var3 * pi * x[2])",
            degree=3,
            var1=mu[0], var2=mu[1], var3=mu[2], pi=self.pi,
        )
        return exact_exp
    
    def exact_solution(self, mu):
        u = fe.Function(self.V)
        u.interpolate(self._exact_exp(mu))
        u_num = np.array(u.vector().get_local())
        self._real_sol = u_num
        return u_num, u
    
    @staticmethod
    def compute_error(approx, exact, text=True):
        err_a = np.linalg.norm(approx-exact)
        err_r = err_a / np.linalg.norm(exact)
        if text:
            print(f"Errore Assoluto: {err_a:.4e}")
            print(f"Errore Relativo: {err_r:.4e}")
        return err_a, err_r
    
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