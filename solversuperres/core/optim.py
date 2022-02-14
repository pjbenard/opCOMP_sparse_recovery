import numpy as np
import numpy.linalg as npl
from tqdm import tqdm

from .linear_operator import Adelta, Adeltap

__all__ = ['generator', 'conjugate_gradient', 'optimal_gradient_descent', 
           'lin_search', 'GD']

def generator(lim=None):
    """
    Acts as an infinite `for` loop

    Yields
    -------
    NoneType 

    Other Parameters
    ----------------
    lim : NoneType, int
        If not None, sets a limit for the generator.
    """
    
    if lim is None:
        i = 0
        while True:
            yield i
            i += 1
            
    else:
        for i in range(lim):
            yield i
    
def conjugate_gradient(A, b, x=None, tol=1e-5, disable_tqdm=True):
    """
    Uses the conjugate gradient method to solve `Ax = b`

    Parameters
    ----------
    A : array_like, shape(`n`, `n`)
        A real or complex hermitian, positive-definite matrix.
    b : array_like, shape(`n`)
        A vector which is the right hand side of the system.

    Returns
    -------
    array_like, shape(`n`)
        The solution of the system `Ax = b`.
        
    Other Parameters
    ----------------
    x : array_like, shape(`n`)
        A starting guess for the solution.
    tol : float
        The maximum tolerance at which the computation may stop.
    disable_tqdm : bool
        Condition if the tqdm CLI interface is displayed during loop.
    """
    
    n = np.size(b)
    x = np.zeros(n) if x is None else x
    
    r = b - A.dot(x)
    p = np.copy(r)
    rsold = r.dot(np.conj(r))
    
    
    for i in tqdm(range(2 * n), disable=disable_tqdm):
    # for _ in tqdm(generator(), disable=disable_tqdm):
        Ap = A.dot(p)
        alpha = rsold / p.dot(np.conj(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(np.conj(r))
        
        if np.sqrt(rsnew) < tol:
            break
            
        p = r + (rsnew / rsold) * p
        rsold = np.copy(rsnew)
        
    return np.real(x)

def optimal_gradient_descent(A, b, x=None, tol=1e-5, disable_tqdm=True):
    """
    Uses the optimal gradient descent method to solve `Ax = b`

    Parameters
    ----------
    A : array_like, shape(`m`, `n`)
        A real or complex matrix.
    b : array_like, shape(`m`)
        A vector which is the right hand side of the system.

    Returns
    -------
    array_like, shape(`n`)
        The solution of the system `Ax = b`.
        
    Other Parameters
    ----------------
    x : array_like, shape(`n`)
        A starting guess for the solution.
    tol : float
        The maximum tolerance at which the computation may stop.
    disable_tqdm : bool
        Condition if the tqdm CLI interface is displayed during loop.
    """
    
    n = np.shape(A)[1]
    x = np.zeros(n) if x is None else x
    
    x_old = np.copy(x)
    for _ in tqdm(generator(), disable=disable_tqdm):
        gradJ = A.T.dot(np.conj(A.dot(np.conj(x_old)) - b))
        # omega = gradJ / 2
        tau = (npl.norm(gradJ)**2) / (2 * npl.norm(A.dot(np.conj(gradJ)))**2)
        x_new = x_old - tau * gradJ
        
        if npl.norm(x_old - x_new) < tol:
            break
            
        x_old = np.copy(x_new)

    return np.real(x_new)

def lin_search(w, r, t, df_dt, tau):
    """
    Find the 'best' next iteration of t using a rough linear search in power of 10

    Parameters
    ----------
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    r : array_like, shape(`m`)
        The residue obtained by `y - Ax_est`.
    t : array_like, shape(`d`)
        A starting guess for the solution.
    df_dt : array_like, shape(`d`)
        The partial derivatives of <A \delta_t, r> by t.
    tau : dict
        Defines the min log(step) and max log(step) to be tested.

    Returns
    -------
    array_like, shape(`d`)
        The best t found during the iteration.
    """
    
    found_tau = False
    expval = Adelta(w, t)
    err = npl.norm(r - expval)

    for etau in range(tau['max'], tau['min'], -1):
        step = 10**etau
        t_int = t - step * df_dt

        expval = Adelta(w, t_int)
        err_int = npl.norm(r - expval)

        if err_int < err:
            err = err_int
            t_best = np.copy(t_int)
            best_tau = etau
            found_tau = True

    if not found_tau:
        t_best = np.copy(t_int)
        
    return t_best

def GD(w, r, t=None, tau=0.01, tol=1e-5, nit=100):
    """
    Uses the gradient descent method to maximise `< A \delta_t, r>`

    Parameters
    ----------
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    r : array_like, shape(`m`)
        The residue obtained by `y - Ax_est`.

    Returns
    -------
    array_like, shape(`d`)
        The maximizer of `< A \delta_t, r>`.
        
    Other Parameters
    ----------------
    t : array_like, shape(`d`)
        A starting guess for the solution.
    tau : {dict, float, int}
        dict = {'min': tau_min, 'max': tau_max} : The range of the descent step.
        float, int : The step of the descent.
    tol : float
        The maximum tolerance at which the computation may stop.
    nit : int
        The number of maximum iterations.
    """
    
    d = np.shape(w)[1]
    t = np.zeros(d) if t is None else t
    
    t_est_0 = np.copy(t)
    
    for it in range(nit):
        df_dt = np.real(np.dot(Adeltap(w, t_est_0).T, np.conj(r)))

        if type(tau) is dict:
            # Linesearch of best tau
            t_est_1 = lin_search(w, r, t_est_0, df_dt, tau)
            
            
        elif type(tau) in (float, int):
            t_est_1 = t_est_0 - tau * df_dt
            
        if npl.norm(t_est_0 - t_est_1) < tol:
            break
        
        t_est_0 = np.copy(t_est_1)
        
    return t_est_1

    