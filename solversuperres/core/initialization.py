import numpy as np
import numpy.linalg as npl
from tqdm import tqdm
import math

from .linear_operator import Adelta, Ax, Aty
from .optim import generator, conjugate_gradient, optimal_gradient_descent, GD
from .glob_optim import gradient_descent

__all__ = ['greedy_spectral', 'trial_nD', 'SCOMP', 'COMP', 'random_points', 'grid_points', 'fixed_points']

def greedy_spectral(y, w, k_init, grid):
    """
    Initialize the positions and amplitudes using a greedy approach

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    k_init : int
        The number of spikes to initialize.
    grid : array_like, shape(`n1`, ..., `nd`, d)
        Coordinates of each point on the grid.

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    """
    
    z = Aty(y, w, grid)
    s = np.shape(z)
    
    idx_sort = np.argsort(z, axis=None)[::-1]
    idx_sort = np.unravel_index(idx_sort, s)
    z_sort = z[idx_sort]

    a_init = z_sort[:k_init]
    t_init = grid[idx_sort][:k_init]
    
    return a_init, t_init
    
def trial_nD(t_true, k_init, dist, opt):
    """
    Initialize the positions around the true positions

    Parameters
    ----------
    t_true : array_like, shape(k_true, d)
        The positions of the true spikes.
    k_init : int
        The number of spikes to initialize.
    dist : float
        The maximal distance of the initialized points from the true points.
    opt : str, ['max_', ''] + 'fixed_distance_`d`D'
        Option on how the initialized positions are constructed.

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    """
    
    d = np.shape(t_true)[1]
    
    a_init = np.ones(k_init)
    t_init = np.zeros((k_init, d))
    
    if opt.startswith('max_fixed_distance'):
        distances = np.random.rand(k_init) * dist
    elif opt.startswith('fixed_distance'):
        distances = np.ones(k_init) * dist
        
    if opt.endswith('1D'):
        X = distances
        dim_coords = [X]
        
    elif opt.endswith('2D'):
        thetas = np.random.rand(k_init) * 2 * np.pi
        
        X = distances * np.cos(thetas)
        Y = distances * np.sin(thetas)
        dim_coords = [X, Y]
        
    elif opt.endswith('3D'):
        phis = np.random.rand(k_init) * 2 * np.pi
        thetas = np.random.rand(k_init) * np.pi
        
        X = distances * np.sin(thetas) * np.cos(phis)
        Y = distances * np.sin(thetas) * np.sin(phis)
        Z = distances * np.cos(thetas)
        dim_coords = [X, Y, Z]
        
    for dim, dim_coord in enumerate(dim_coords):
        t_init[:, dim] = dim_coord + t_true[:, dim]
        
    return a_init, t_init
        
def SCOMP(y, w, tau, opt='no_ernergy_loss', minimizer='ogd', nb_tests=1, k_init=None, min_iter=None):
    """
    Initialize the positions and amplitudes using the SCOMP algorithm

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    tau : {dict, float, int}
        dict = {'min': tau_min, 'max': tau_max} : The range of the descent step.
        float, int : The step of the descent.

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    matrix_condition : array_like, shape(k_init)
        The condition number of the intermediate matrix at each iterations.
    errors : array_like, shape(k_init+1)
        The approximation error between the observation `y` and the observed estimated signal `x_est` at each iterations.
    residue : array_like, shape(`m`)
        The remaining residue obtain from the difference between the observation and the estimated signal.
        
    Other Parameters
    ----------------
    opt : str
        Chooses the exit condition for the initialization method.
        `no_ernergy_loss` : Exit when the ernergy of the system doesn't decreases.
        `fixed_init_spikes` : Exit when `k_init` spikes are initialized.
    minimizer : str
        Chooses the minimization algorithm for the amplitudes.
        `lstsq` : The blackbox least-square algorithm by Numpy.
        `cg' : The Conjugate Gradient method.
        `ogd` : The Optimal Gradient Descent Method.
    k_init : NoneType, int
        Forces the number of spikes to initialize.
        If opt is `fixed_init_spikes`, k_init must be fixed.
    min_iter : NoneType, int
        The lowest iteration at which the method can stop.
    """
    
    lim_loop = None if opt == 'no_ernergy_loss' else k_init
    
    d = np.shape(w)[1]
    
    r = np.copy(y)
    T = np.array([]).reshape(0, d)
    a_best = np.array([])

    errors = []
    errors.append(npl.norm(r))

    matrix_condition = []

    for i in tqdm(generator(lim=lim_loop)):
        t_best = GD(w=w, r=r, t=np.random.rand(d), tau=tau)
        
        # Tries multiple random initialization to find the best
        r_best = npl.norm(r - Adelta(w, t_best))
        for test in range(1, nb_tests):
            t_int = GD(w=w, r=r, t=np.random.rand(d), tau=tau)
            r_int = npl.norm(r - Adelta(w, t_int))
            if r_int < r_best:
                t_best = np.copy(t_int)
                r_best = r_int

        T = np.concatenate((T, t_best[np.newaxis, :]), axis=0)
        a_prev = np.concatenate((a_best, [0]))

        # M = np.zeros((m, i+1), dtype=complex)
        # for j in range(i+1):
        #     M[:, j] = Adelta(np.dot(w, T[j]))
        
        M = Adelta(w, T)

        matrix_condition.append(npl.cond(M))

        if minimizer == 'lstsq':
            a_best = npl.lstsq(M, y, rcond=None)[0]
        elif minimizer == 'cg':
            MtM = M.T.dot(np.conj(M))
            Mty = M.T.dot(np.conj(y))
            a_best = conjugate_gradient(MtM, Mty, a_prev, tol=1e-10)
        elif minimizer == 'ogd':
            a_best = optimal_gradient_descent(M, y, a_prev, tol=1e-10)
        
        r = y - np.dot(M, a_best)
        errors.append(npl.norm(r))

        if lim_loop is None and (min_iter is None or i > min_iter):
            if errors[-1] > errors[-2] * 1.02:
                T = T[:-1]
                a_best = a_prev[:-1]
                break

            if np.isclose(errors[-1], errors[-2]):
                break
                
    a_init = np.copy(a_best)
    t_init = np.copy(T)
    
    return a_init, t_init, np.array(matrix_condition), np.array(errors), r
   
    
def COMP(y, w, tau, minimizer='ogd', nb_tests=1, k_init=None, **GDkwargs):
    """
    Initialize the positions and amplitudes using the COMP algorithm

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    k_init : int
        The number of spikes to initialize.
    tau : {dict, float, int}
        dict = {'min': tau_min, 'max': tau_max} : The range of the descent step.
        float, int : The step of the descent.

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    matrix_condition : array_like, shape(k_init)
        The condition number of the intermediate matrix at each iterations.
    errors : array_like, shape(k_init+1)
        The approximation error between the observation `y` and the observed estimated signal `x_est` at each iterations.
    residue : array_like, shape(`m`)
        The remaining residue obtain from the difference between the observation and the estimated signal.
        
    Other Parameters
    ----------------
    minimizer : str
        Chooses the minimization algorithm for the amplitudes.
        `lstsq` : The blackbox least-square algorithm by Numpy.
        `cg` : The Conjugate Gradient method.
        `ogd` : The Optimal Gradient Descent Method.
    GDkwargs : dict
        Include parameters specific to the gradient descent method.
        tau : {dict, float, int}
            dict = {'min': tau_min, 'max': tau_max} : The range of the descent step.
        nit : int
            The number of maximum iterations.
    """
    
    lim_loop = k_init
    
    d = np.shape(w)[1]
    
    r = np.copy(y)
    T = np.array([]).reshape(0, d)
    a_best = np.array([])

    errors = []
    errors.append(npl.norm(r))
    
    matrix_condition = []

    for i in tqdm(generator(lim=lim_loop), disable=True):
        t_best = GD(w=w, r=r, t=np.random.rand(d), tau=tau)
        
        # Tries multiple random initialization to find the best
        r_best = npl.norm(r - Adelta(w, t_best))
        for test in range(1, nb_tests):
            t_int = GD(w=w, r=r, t=np.random.rand(d), tau=tau)
            r_int = npl.norm(r - Adelta(w, t_int))
            if r_int < r_best:
                t_best = np.copy(t_int)
                r_best = r_int

        T = np.concatenate((T, t_best[np.newaxis, :]), axis=0)
        a_prev = np.concatenate((a_best, [0]))

        M = Adelta(w, T)
        
        matrix_condition.append(npl.cond(M))
            
        if minimizer == 'lstsq':
            a_best = npl.lstsq(M, y, rcond=None)[0]
        elif minimizer == 'cg':
            MtM = M.T.dot(np.conj(M))
            Mty = M.T.dot(np.conj(y))
            a_best = conjugate_gradient(MtM, Mty, a_prev, tol=1e-10)
        elif minimizer == 'ogd':
            a_best = optimal_gradient_descent(M, y, a_prev, tol=1e-10)

        a_best, T, _, _, _ = gradient_descent(y, w, a_best, T, project=False, **GDkwargs)
        
        r = y - Ax(a_best, w, T)
        # r = y - np.dot(M, a_best)
        errors.append(npl.norm(r))
        
        if (lim_loop is None) and (errors[-1] < 1e-3):
            break

    a_init = np.copy(a_best)
    t_init = np.copy(T)
    
    return a_init, t_init, np.array(matrix_condition), np.array(errors), r
    
    
def random_points(k_init, d):
    """
    Initialize the positions and amplitudes randomly on the domain [0, 1] x [0, 1]

    Parameters
    ----------
    k_init : int
        The number of spikes to initialize.
    d : int
        The number of dimensions for the positions

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    """
    
    a_init = np.random.rand(k_init)
    t_init = np.random.rand(k_init, d)
    
    return a_init, t_init
    
    
def grid_points(eps_dist_spikes, d):
    """
    Initialize the positions and amplitudes on the grid [0, 1] x [0, 1]

    Parameters
    ----------
    eps_dist_spikes : float
        The minimal distance between two spikes of the signal `x_0`.
    d : int
        The number of dimensions for the positions.

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    """
    N = math.ceil(1 / eps_dist_spikes)
    
    arange = np.arange(0, N+1) / N
    Xs = np.meshgrid(*[arange] * d)
    
    t_init = np.stack(Xs, axis=-1).reshape(-1, d)
    a_init = np.ones(t_init.shape[0]) / N**(1/4)
    
    return a_init, t_init
    
    
def fixed_points(a_fixed, t_fixed):
    """
    Initialize the positions and amplitudes using user-defined amplitudes and positions

    Parameters
    ----------
    a_fixed : array_like, shape(`k_init`)
        The user-fixed amplitudes.
    t_fixed : array_like, shape(`k_init`, d`)

    Returns
    -------
    a_init : array_like, shape(`k_init`)
        The amplitudes of the initalized spikes.
    t_init : array_like, shape(`k_init`, `d`)
        The positions of the initalized spikes.
    """
    
    a_init = np.copy(a_fixed)
    t_init = np.copy(t_fixed)
    
    return a_init, t_init
