import holoviews as hv

__all__ =  ['error', 'condition']

opts_curve = dict(linewidth=1, fig_size=300, aspect=2.5)

def _curve(data, options):
    logy = options.get('logy', False)
    ylim = (0.9 * min(data), 1.1 * max(data)) if logy else (0, 1.1 * max(data))
    
    plot = hv.Curve(data).opts(**options, ylim=ylim)
    
    return plot

def error(solver, alg='COMP', **options):
    err_arg = f'{alg}_error'
    data = getattr(solver, err_arg)
    
    if alg == 'COMP':
        xlabel = 'Sliding COMP iterations'
    elif alg == 'SCOMP':
        xlabel = 'Over-parametrized COMP iterations'
    elif alg == 'PGD':
        xlabel = 'Projected Gradient Descent iterations'
    else:
        xlabel = 'Iterations'
    
    error_options = dict(xlabel=xlabel, ylabel=r'Norm of residue $\|r\, \|$')
    
    plot = _curve(data, {**opts_curve, **error_options, **options})
    
    return plot

def condition(solver, alg='COMP', **options):
    cond_arg = f'{alg}_matrix_condition'
    data = getattr(solver, cond_arg)
    
    if alg == 'COMP':
        xlabel = 'Sliding COMP iterations'
    elif alg == 'SCOMP':
        xlabel = 'Over-parametrized COMP iterations'
    else:
        xlabel = 'Iterations'
    
    cond_options = dict(xlabel=xlabel, ylabel='Conditon number')
    
    plot = _curve(data, {**opts_curve, **cond_options, **options})
    
    return plot
