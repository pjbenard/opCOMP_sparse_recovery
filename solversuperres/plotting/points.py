import numpy as np
import holoviews as hv

__all__ = ['true_points', 'esti_points']

color_range = np.arange(255, -1, -1)

red_cmap = list(map(lambda c: f'#ff{c:02x}{c:02x}', color_range))
green_cmap = list(map(lambda c: f'#{c:02x}ff{c:02x}', color_range))
blue_cmap = list(map(lambda c: f'#{c:02x}{c:02x}ff', color_range))

diverging_cmap = green_cmap[::-1] + blue_cmap[1:]

true_cmap = red_cmap
esti_cmap = blue_cmap

# opts_points = dict(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
opts_points = dict(xlim=(0, 1), ylim=(0, 1), fig_size=300, aspect=1, s=300, edgecolor=None)

def filter_in_bounds(array, bounds=[(0, 1), (0, 1)]):
    carray = np.copy(array)
    for dim, (lb, ub) in enumerate(bounds):
        carray = carray[np.logical_and(lb <= carray[:, dim], carray[:, dim] <= ub)]
    
    return carray

def get_cmap(data, cmap_pos, cmap_sym):
    s_a = data[:, -1]
    if np.all(s_a > 0):
        cmap = cmap_pos
        symmetric = False
        clim = (0, max(s_a))
    else:
        cmap = cmap_sym
        symmetric = True
        lim = max(abs(s_a))
        clim = (-lim, lim)
        
    return cmap, symmetric, clim

def _points(data, options):
    plot = hv.Points(data, vdims=['amps']).opts(color='amps', clabel='Amplitude', **options)
    
    return plot

def true_points(signal, a='a', t='t', **options):
    s_a = getattr(signal, a)
    s_t = getattr(signal, t)
    
    data = np.concatenate((s_t, s_a[:, None]), axis=-1)
    data = filter_in_bounds(data)
    
    cmap, symmetric, clim = get_cmap(data, true_cmap, 'RdGy')
    
    s_true_dict = dict(marker='x', cmap=cmap, symmetric=symmetric, clim=clim, colorbar=True)
    # s_true_dict = dict(marker='X', colorbar=True)
    s_true_options = {**opts_points, **s_true_dict, **options}
    
    plot = _points(data, s_true_options).relabel('Ground Truth')
    # plot = plot.redim.range(amps=vdims_range)
    
    return plot

def esti_points(signal, a='a', t='t', **options):
    s_a = getattr(signal, a)
    s_t = getattr(signal, t)
    
    data = np.concatenate((s_t, s_a[:, None]), axis=-1)
    data = filter_in_bounds(data)
    
    cmap, symmetric, clim = get_cmap(data, esti_cmap, diverging_cmap)
        
    s_esti_dict = dict(marker='+', cmap=cmap, symmetric=symmetric, clim=clim, colorbar=True)
    # s_true_dict = dict(marker='o', colorbar=True)
    s_esti_options = {**opts_points, **s_esti_dict, **options}
    
    plot = _points(data, s_esti_options).relabel('Estimation')
    # plot = plot.redim.range(amps=vdims_range)
    
    return plot