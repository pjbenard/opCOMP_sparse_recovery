import numpy as np

from solversuperres import core

__all__ = ['Signal_true', 'Signal_esti']

class Signal(object):
    def __init__(self, d=2, k=5, m=1, eps_dist_spikes=0.1):
        self.d = d
        self.k = k
        self.m = m
        self.eps_dist_spikes = eps_dist_spikes
        
    
class Signal_true(Signal):
    def __init__(self, d=2, k=5, m=1, eps_dist_spikes=0.1):
        super().__init__(d=d, k=k, m=m, eps_dist_spikes=eps_dist_spikes)
    
    def init_spikes(self, manual_a=None, manual_t=None, simplify=True, 
                    force_nb_spikes=True, force_amplitudes=True):
        first_it = True
        while first_it or force_nb_spikes:
            self.a = np.ones(self.k) if manual_a is None else manual_a
            self.t = np.random.rand(self.k, self.d) if manual_t is None else manual_t
        
            # Project spikes too close
            if simplify or force_nb_spikes:
                self.a, self.t = core.project_theta_eps(self.a, self.t, self.eps_dist_spikes)
            
            first_it = False
            # Stay true if number of spikes after projection is lower than before
            force_nb_spikes &= np.size(self.a) != self.k
            
        self.k = np.size(self.a)
            
        if force_amplitudes:
            self.a = np.ones(self.k) if manual_a is None else manual_a[:self.k]
            
    def init_mesures(self, c=0.1):
        self.w = np.random.randn(self.m, self.d) / c
        self.y = core.Ax(self.a, self.w, self.t)
        
    def create_estimation(self, coef_k=5):
        signal_esti = Signal_esti(d=self.d, k=self.k * coef_k, m=self.m, eps_dist_spikes=self.eps_dist_spikes)
        
        return signal_esti


class Signal_esti(Signal):
    def __init__(self, d=2, k=5, m=1, eps_dist_spikes=0.1):
        super().__init__(d=d, k=k, m=m, eps_dist_spikes=eps_dist_spikes)
        
    
