import numpy as np

from solversuperres import core

__all__ = ['Solver']

class Solver(object):
    def __init__(self, s_true, s_esti=None, coef_k=5):
        self.s_true = s_true
        self.s_esti = s_true.create_estimation(coef_k=coef_k) if s_esti is None else s_esti
        
    def initialize(self, opt, dyn_plot=None, **kwargs):
        init_matrix_condition = None
        init_error = None
        
        if opt == 'greedy_spectral':
            a_esti, t_esti = core.greedy_spectral(y=self.s_true.y, w=self.s_true.w, k_init=self.s_esti.k, **kwargs)
            
        elif opt == 'trial_nD':
            a_esti, t_esti = core.trial_nD(t_true=self.s_true.t, k_init=self.s_esti.k, **kwargs)
        
        elif opt == 'SCOMP':
            a_esti, t_esti, *other_params = core.SCOMP(y=self.s_true.y, w=self.s_true.w, **kwargs)
            
            init_matrix_condition, init_error, init_residue = other_params
            self.SCOMP_matrix_condition = init_matrix_condition
            self.SCOMP_error = init_error
            self.SCOMP_residue = init_residue
            
        elif opt == 'COMP':
            a_esti, t_esti, *other_params = core.COMP(y=self.s_true.y, w=self.s_true.w, k_init=self.s_true.k, **kwargs)
            
            init_matrix_condition, init_error, init_residue = other_params
            self.COMP_matrix_condition = init_matrix_condition
            self.COMP_error = init_error
            self.COMP_residue = init_residue
            
        elif opt == 'random':
            a_esti, t_esti = core.random_points(self.s_esti.k, self.s_true.d)
            
        elif opt == 'grid':
            a_esti, t_esti = core.grid_points(self.s_true.eps_dist_spikes, self.s_true.d)
            
        elif opt == 'fixed_points':
            a_esti, t_esti = core.fixed_points(**kwargs)
            
        else:
            a_esti = np.zeros(self.s_esti.k)
            t_esti = np.zeros((self.s_esti.k, self.s_esti.d))
            
        self.s_esti.a_init = a_esti
        self.s_esti.t_init = t_esti
        
        self.s_esti.k = np.size(a_esti)
                
        self.s_esti.a = np.copy(self.s_esti.a_init)
        self.s_esti.t = np.copy(self.s_esti.t_init)
        
    def optimize(self, **kwargs):
        a_esti, t_esti, traj_a, traj_t, opt_error = core.gradient_descent(y=self.s_true.y, w=self.s_true.w, a_init=self.s_esti.a_init, t_init=self.s_esti.t_init, **kwargs)
        
        self.s_esti.a_opt = a_esti
        self.s_esti.t_opt = t_esti
        self.s_esti.traj_a = traj_a
        self.s_esti.traj_t = traj_t
        self.PGD_error = opt_error
        
        self.s_esti.a = np.copy(self.s_esti.a_opt)
        self.s_esti.t = np.copy(self.s_esti.t_opt)