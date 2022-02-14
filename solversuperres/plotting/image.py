import numpy as np
import holoviews as hv

__all__ =  ['energy2D']

opts_image = dict(bounds=(0, 0, 1, 1), colorbar=True, fig_size=300, aspect=1)

def energy2D(data, **options):
    bounds = opts_image.pop('bounds', (0, 0, 1, 1))
    plot = hv.Image(np.flipud(data), bounds=bounds).opts(**opts_image, **options)
    
    return plot