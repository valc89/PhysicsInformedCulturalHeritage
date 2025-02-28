import numpy as np
import fenics as fe

class HeatFEM(object):

    def __init__(self, mesh, time_interval, data : np.ndarray):
        self.mesh = mesh
        self.data = data
        self.n_mesh_p = mesh.coordinates().shape[0]
        self.time_interval = time_interval
        self.temp_points = self.data.shape[0]
        self.V = self._initialize_problem()
        self.inner_product = self._inner_product()
        self._sol = None
    
    def _initialize_problem(self):
        V_el = fe.FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
        V = fe.FunctionSpace(self.mesh, V_el)
        return V
    
    def _inner_product(self):
        u = fe.TestFunction(self.V)
        v = fe.TrialFunction(self.V)
        return fe.assemble(fe.inner(fe.grad(u), fe.grad(v))*fe.dx)
    
    def _initialize_fem_problem(self, t_0, T):
        if abs(t_0-0) < np.finfo(np.float64).eps:
            h = T / (self.temp_points-1)
            # Relativo alle condizioni iniziali
            u_n = fe.interpolate(fe.Constant(self.data[0]), self.V)
            initial = np.array(u_n.vector().get_local())
            # Memorizzare soluzione
            solution_matrix = np.zeros((self.n_mesh_p, self.temp_points))
            solution_matrix[:, 0] = initial[:self.n_mesh_p]
            return h, u_n, solution_matrix
        else:
            raise NotImplementedError("Only t_0 = 0")
    
    def _problem_forms(self, u_n, h):
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        # Definizione della forma bilineare e lineare per l'equazione del calore
        a = u*v*fe.dx + h*fe.dot(fe.grad(u), fe.grad(v))*fe.dx
        L = u_n * v * fe.dx
        return a, L
    
    def _single_step(self, step, h, u_n):
        bc = fe.DirichletBC(self.V, fe.Constant(self.data[step]), fe.DomainBoundary())
        bcs = []
        bcs.append(bc)
        a, L = self._problem_forms(u_n, h)
        A, b = fe.assemble_system(a, L, bcs)
        solution = fe.Function(self.V)
        fe.solve(A, solution.vector(), b)
        return solution

    def solve_fem(self):
        t_0, T = self.time_interval
        h, u_n, solution_matrix = self._initialize_fem_problem(t_0, T)
        lst_solutions = []
        for step in range(1, self.temp_points):
            print(f"Solving at step {step}")
            sol = self._single_step(step, h, u_n)
            num_sol = np.array(sol.vector().get_local())
            solution_matrix[:, step] = num_sol[:self.n_mesh_p]
            u_n.assign(sol)
            lst_solutions.append(sol)
        return solution_matrix, lst_solutions
    
    @property
    def sol(self):
        if self._sol is None:
            raise AttributeError("No solution found. Call the method solve_fem.")
        return self._sol