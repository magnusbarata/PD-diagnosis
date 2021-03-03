import os
import json
import pandas as pd

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
    for i, fname in enumerate(fnames):
        label = 1 if fname[:2] == 'PD' else 0
        fnames[i] = f'{dirpath}/{fname}'
        df['Label'].append(label)
    
    pd.DataFrame(df).to_csv(fout, index=False)