from .baselaplace import BaseLaplace

def Laplace(model,
            likelihood, 
            subset_of_weights='all',
            hessian_structure='full', 
            *args, 
            **kwargs,
            ): 
    """
    Simplified Laplace access using strings instead of different classes.
    Args:
    subset_of_weights: {'last_layer', 'subnetwork', 'all'}, subset of weights to consider for inference.
    hessian_structure: {'diag', 'kron', 'full'}, structure of the Hessian approcximation.
    """
    laplace_map = {subclass._key: subclass for subclass in _all_subclasses(BaseLaplace) if hasattr(subclass, '_key')}
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)] # access subclass with keys. 
    return laplace_class(model, likelihood, *args, **kwargs)
    
    
def _all_subclasses(cls): 
    """recursively find all the subclasses of cls. """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])
    