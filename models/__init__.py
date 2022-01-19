import importlib
import functools

def find_model(model_name):
    """Import a model using model name"""
    if 'predefined_' in model_name:
        model_name = model_name.split('_')[-1]
        fname = f'tensorflow.keras.applications'
        is_predefined = True
    else:
        fname = f'models.{model_name}'
        is_predefined = False
    
    modellib = importlib.import_module(fname)
    model_fn = getattr(modellib, f'{model_name}')
    if is_predefined:
        model_fn = functools.partial(model_fn, weights=None)
    return model_fn