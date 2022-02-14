import numpy as np
import numpy.linalg as npl

__all__ = ['Adelta', 'Adeltap', 'alphaf', 'alphafp', 
           'Ax', 'Aty']

def Adelta(w, t):
    """
    The observation of delta_t through `A`

    Parameters
    ----------
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    t : array_like, shape(`k` ,`d`)
        The positions of the spikes.

    Returns
    -------
    array_like, shape(`m`, `k`)
        Each column represents the observations of each spike through `A`.
    """

    m = np.shape(w)[0]
    y = alphaf(w, t)
    
    return y / m**.5

def Adeltap(w, t):
    """
    Compute the partial derivatives of `Adelta`

    Parameters
    ----------
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    t : array_like, shape(`d`)
        The positions of the spikes.

    Returns
    -------
    array_like, shape(`m`)
        The patrial derivatives.
    """
    
    m = np.shape(w)[0]
    y = - np.real(alphafp(w, t))
    
    return y / m**.5
    
def alphaf(w, t):
    """
    Compute the exponential of the dot product between the measurements and positions

    Parameters
    ----------
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    t : array_like, shape(`k` ,`d`)
        The positions of the spikes.

    Returns
    -------
    array_like, shape(`m`, `k`)
        Each column represents exponential of the dot product between the measurements and each spike.
    """
    
    wt = np.dot(w, t.T)
    
    return np.exp(1j * wt)
    
def alphafp(w, t):
    """
    Compute the partial derivatives of `alphaf`
    
    Parameters
    ----------
    w : array_like, shape(m, d)
        The list of measurements in `A`.
    t : array_like, shape(d)
        Position of the sipke.
        
    Returns
    -------
    array_like, shape(`m`, `k`)
        The partial derivative of `alphaf` at each position `t`.
    """
    
    return 1j * w * alphaf(w, t)[:, None]
    
def Ax(a, w, t):
    """
    The observation of delta_t through `A` with amplitudes

    Parameters
    ----------
    a : array_like, shape(`k`)
        The amplitudes of the spikes.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    t : array_like, shape(`k` ,`d`)
        The positions of the spikes.

    Returns
    -------
    array_like, shape(`m`)
        The observations of the spikes with amplitude through `A`.
    """
    
    expval = Adelta(w, t)
        
    return np.dot(expval, a)

def Aty(y, w, grid):
    """
    Compute the "pseudo-inverse" of the problem A \delta_t = y for each point on the grid

    Parameters
    ----------
    y : array_like, shape(`m`)
        The observation of the signal `x_0`.
    w : array_like, shape(`m`, `d`)
        The list of measurements in `A`.
    grid : array_like, shape(`n1`, ..., `nd`, d)
        Coordinates of each point on the grid.

    Returns
    -------
    array_like, shape(`n1`, ..., `nd`)
        Each element of the array represent the value of the "pseudo-inverse" `Aty`
    """
    
    m = np.shape(w)[0]
    *dims_size, d = np.shape(grid)
        
    expval = alphaf(w, -grid.reshape((-1, d)))
    
    z = np.dot(expval.T, y).reshape(dims_size)       
    z = npl.norm(z[..., np.newaxis], axis=-1)
 
    return z / dims_size[0]
