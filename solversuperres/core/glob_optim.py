import numpy as np
import numpy.linalg as npl
from tqdm import tqdm

from .linear_operator import Adelta, Ax

__all__ = ['project_theta_eps', 'Nablag', 'norm_residue', 'gradient_descent']

def project_theta_eps(a, t, eps_proj, cut_off=1e-3):
    """
    Project the parameter theta into a subspace. 
    Merges the spikes which are too close from each others.

    Parameters
    ----------
    a : array_like, shape(`k0`)
        The amplitudes of the spikes.
    t : array_like, shape(`k0`, `d`)
        The positions of the spikes.
    eps_proj : float
        The distance at which two spikes are considered too close and are merged.

    Returns
    -------
    a_out : array_like, shape(`k1`)
        The amplitudes of the projected spikes.
    t_out : array_like, shape(`k1`, `d`)
        The positions of the projected spikes.
    """
    k, d = np.shape(t)
    a_temp = np.copy(a)
    t_temp = np.copy(t)
        
    for i in range(0, k):
        if a_temp[i] == 0:
            continue
        elif abs(a_temp[i]) < cut_off:
            a_temp[i] = 0
            t_temp[i] = 0
            continue
        for j in range(i + 1, k):
            if a_temp[j] == 0:
                continue
            dt = t_temp[i] - t_temp[j]
            if npl.norm(dt, ord=2) < eps_proj:
                c1 = abs(a_temp[i])
                c2 = abs(a_temp[j])
                
                t_temp[i] = (c1 * t_temp[i] + c2 * t_temp[j]) / (c1 + c2)
                a_temp[i] += a_temp[j]
                a_temp[j] = 0
                t_temp[j] = 0
        
    
    a_out = a_temp[a_temp != 0]
    t_out = t_temp[a_temp != 0]
    
    return a_out, t_out

def Nablag(y, w, a, t, coord):
    """
    Compute the partial derivative of `g = ||A phi(a, t) - y||`.

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    a : array_like, shape(`k`)
        The amplitudes of the estimated spikes.
    t : array_like, shape(`k`, `d`)
        The positions of the estimated spikes.
    coord : int
        Determine if it compute the partial derivative in `a` or `t`.
        If coord == 1 : partial derivative in `a`.
        Else : partial derivative in `t`.

    Returns
    -------
    da : array_like, shape(`k`)
        The partial derivative of `g` in `a`.
    dt : array_like, shape(`k`, `d`)
        The partial derivative of `g` in `t`.
    """
    partial_da = coord == 1
    
    m, d = np.shape(w)
    k = np.shape(t)[0]
    
    da = np.zeros(k)
    dt = np.zeros((k, d))

    expval = Adelta(w, t)    
    res = Ax(a, w, t) - y
    
    if partial_da:
        da = 2 * np.real(np.dot(expval.T, np.conj(res)))
        
    else:
        dexpval_dt = 1j * w[..., None] * expval[:, None, :]
        dt = 2 * a[:, None] * np.real(np.dot(dexpval_dt.T, np.conj(res)))
        
    return da, dt

def norm_residue(y, w, a, t):
    """
    Compute the norm of the residue.

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    a : array_like, shape(`k`)
        The amplitudes of the estimated spikes.
    t : array_like, shape(`k`, `d`)
        The positions of the estimated spikes.

    Returns
    -------
    float
        The norm of the residue.
    """
    
    residue = Ax(a, w, t) - y
    
    return npl.norm(residue, ord=2)

def gradient_descent(y, w, a_init, t_init, eps_proj=0.05, 
                     project=True, tau={'min': -5, 'max': 0}, nit=500):
    """
    Uses the gradient descent method to maximise minimize `g`.

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    a : array_like, shape(`k0`)
        The amplitudes of the initialized spikes.
    t : array_like, shape(`k0`, `d`)
        The positions of the initialized spikes.

    Returns
    -------
    a_est : array_like, shape(`k1`)
        The estimated amplitudes that minimize `g`.
    t_est : array_like, shape(`k1`)
        The estimated positions that minimize `g`.
    traj_a : array_like, shape(`nit`, `k`)
        The evolution of the amplitudes throughout the iterations.
    traj_a : array_like, shape(`nit`, `k`)
        The evolution of the positions throughout the iterations.
        
    Other Parameters
    ----------------
    eps_proj : float
        The distance at which two spikes are considered too close and are merged.
    project : bool
        Apply or not the projection method on spikes too close.
    tau : {dict, float, int}
        dict = {'min': tau_min, 'max': tau_max} : The range of the descent step.
    nit : int
        The number of maximum iterations.
    """
    
    m, d = np.shape(w)
    k = np.shape(t_init)[0]
    k_int = k
    
    a_est = np.copy(a_init)
    t_est = np.copy(t_init)
    
    traj_a = np.zeros((nit, k))
    traj_t = np.zeros((nit, k, d))
    
    errors = []
    
    for it in tqdm(range(nit), desc=f'Nb spikes = {k}'):
        norm_res = norm_residue(y, w, a_est, t_est)
        errors.append(norm_res)
        
        da, dt = Nablag(y, w, a_est, t_est, it % 2)

        for etau in range(tau['max'], tau['min'], -1):
            step = 10**etau
            
            a_est_int = a_est - step * da
            t_est_int = t_est - step * dt    
                
            norm_res_int = norm_residue(y, w, a_est_int, t_est_int)
            
            if norm_res_int < norm_res:
                a_est = np.copy(a_est_int)                
                t_est = np.copy(t_est_int)
                
                norm_res = norm_res_int
                best_etau = etau
                
        if project and (it % 2 == 0):
            a_est, t_est = project_theta_eps(a_est, t_est, eps_proj)
            k_int = np.shape(t_est)[0]
        
        traj_a[it, :k_int] = np.copy(a_est)
        traj_t[it, :k_int] = np.copy(t_est)
        
        # if disp_info:
        #     info = f'Iteration {it+1:>3}, function value = {gval}, last tau = {best_etau}'
        #     print(info)
        
        if abs(norm_res) < k**.5 * 1e-5:
            traj_a = traj_a[:it+1]
            traj_t = traj_t[:it+1]
            break
        
    errors.append(norm_res)
    
    return a_est, t_est, traj_a, traj_t, np.array(errors)