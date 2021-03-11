import argparse
from datetime import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd

from generators.aug_gen import random_noise, random_rotation, random_shift, random_zoom, AugGen
from models.baseline import baseline_model
from utils import *
keras.backend.clear_session()

def main(args):
    params = Params(args.settings)
    df = pd.read_csv(params.data)
    X, y = df.Fpath.values, df.Label.values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, train_size=0.85, random_state=params.seed)
    gen_tr = AugGen(X_tr, y_tr, batch_size=params.batch_size, aug_fns=[random_noise, random_rotation, random_shift, random_zoom])
    gen_val = AugGen(X_val, y_val, batch_size=params.batch_size)
    
    if continue_train_on(args.exp_dir):
        pass # TODO
    else:
        model = baseline_model((256, 256, 1), gen_tr.n_class)
        optimizer = keras.optimizers.Adam(lr=params.lr, decay=params.decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    chkpoint_best = keras.callbacks.ModelCheckpoint(f'{args.exp_dir}/model_best.h5', monitor='val_loss', save_best_only=True)
    chkpoint_latest = keras.callbacks.ModelCheckpoint(f'{args.exp_dir}/model_latest.h5', monitor='loss')
    logger = keras.callbacks.CSVLogger(f'{args.exp_dir}/loss.csv')
    model.fit(gen_tr, validation_data=gen_val,
              epochs=params.n_epochs, verbose=1, use_multiprocessing=False, workers=4, callbacks=[chkpoint_best, chkpoint_latest, logger])
    params.save(f'{args.exp_dir}/train_params.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', nargs='?', default=datetime.today().strftime('%Y%m%d_EXP'))
    parser.add_argument('--settings', default='default_settings.json')
    parser.add_argument('--data_dir', help='Assign a data directory.')
    main(parser.parse_args())