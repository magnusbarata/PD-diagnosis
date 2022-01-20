import os
import re
import json
import shutil
import subprocess
import pandas as pd

def set_seed(seed):
    """Function setting the global seed for reproducible results"""
    import tensorflow as tf
    import numpy as np
    import random

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def limit_gpu_memory(mem_limit, gpu_idx=0):
    """Function to limit GPU memory to be allocated by tf"""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[gpu_idx],
                [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)]
            )
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def get_gpu_name():
    return subprocess.check_output(
        'nvidia-smi --query-gpu=name --format=csv,noheader', shell=True
    ).decode('utf-8')  


def continue_train_on(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        usr_in = input(f'{path} already exists. Overwrite? (y/n): ')
        if usr_in.lower() == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            usr_in = input('Continue training this model? (y/n): ')
            if usr_in.lower() == 'y': return True
            else:
                print('Exiting...')
                raise SystemExit


class Params:
    """Class to load/save hyperparameters from/to a json file.
    
    Usage sample:
    ```
    params = Params(hyperparams.json)
    print(params.epoch)
    params.epochs = 100 # change the number of epochs in params
    ```
    """
    def __init__(self, fpath):
        self.update(fpath)

    def save(self, fpath):
        with open(fpath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        
    def update(self, fpath):
        with open(fpath) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        return self.__dict__


def build_dataset(data_dir, fout='dataset.csv'):
    dirpath, _, fnames = next(os.walk(data_dir))
    df = {'Fpath': fnames, 'Label': []}
    cls_map = {'nor': 0, 'pd': 1} # Basic case is binary classification
    for i, fname in enumerate(fnames):
        label_str = re.match(r'\D+', fname)[0].lower()
        if cls_map.get(label_str) is None:
            cls_map[label_str] = len(cls_map)
        fnames[i] = f'{dirpath}/{fname}'
        df['Label'].append(cls_map[label_str])
    
    pd.DataFrame(df).to_csv(fout, index=False)
    return cls_map
