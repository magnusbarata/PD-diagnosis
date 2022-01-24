import argparse
from datetime import datetime
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd

from generators.aug_gen import random_noise, random_rotation, random_shift, random_zoom, AugGen
from generators.vol_gen import VolGen
from models import find_model
from utils import *
keras.backend.clear_session()

def main(args):
    if args.mem_limit:
        limit_gpu_memory(args.mem_limit)
    
    params = Params(args.settings)
    set_seed(params.seed)
    if args.data_dir:
        params.cls_map = build_dataset(args.data_dir, params.data)
    df = pd.read_csv(params.data)
    X, y = df.Fpath.values, df.Label.values
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, stratify=y, train_size=0.85, random_state=params.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, train_size=0.89, random_state=params.seed)
    datagen_params = params.datagen_params if hasattr(params, 'datagen_params') else {}
    if hasattr(params, 'volume_size'):
        gen_tr = VolGen(X_tr, y_tr, batch_size=params.batch_size, volume_size=params.volume_size, **datagen_params)
        gen_val = VolGen(X_val, y_val, batch_size=params.batch_size, volume_size=params.volume_size, **datagen_params)
    else:
        gen_tr = AugGen(
            X_tr, y_tr,
            batch_size=params.batch_size,
            image_size=params.image_size,
            aug_fns=[random_noise, random_rotation, random_shift, random_zoom],
            gan_generators=[keras.models.load_model(fname) for fname in params.gan_generators],
            **datagen_params
        )
        gen_val = AugGen(X_val, y_val, batch_size=params.batch_size, image_size=params.image_size, **datagen_params)
    
    if continue_train_on(args.exp_dir):
        pass # TODO
    else:
        model_params = {'classes': gen_tr.n_class}
        if hasattr(params, 'model_params'):
            model_params.update(params.model_params)
        model = find_model(params.model)(input_shape=gen_tr.x_shape[1:], **model_params)
        optimizer = keras.optimizers.Adam(lr=params.lr, decay=params.decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    chkpoint_best = keras.callbacks.ModelCheckpoint(f'{args.exp_dir}/model_best.h5', monitor='val_accuracy', save_best_only=True)
    chkpoint_latest = keras.callbacks.ModelCheckpoint(f'{args.exp_dir}/model_latest.h5', monitor='loss')
    logger = keras.callbacks.CSVLogger(f'{args.exp_dir}/loss.csv')
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'{args.exp_dir}/logs', histogram_freq=1)
    stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    model.fit(gen_tr, validation_data=gen_val,
              epochs=params.n_epochs,
              verbose=1,
              use_multiprocessing=False,
              workers=4,
              callbacks=[chkpoint_best, chkpoint_latest, logger, tensorboard, stopper]
    )
    params.gpu_name = get_gpu_name()
    params.save(f'{args.exp_dir}/train_params.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', nargs='?', default=datetime.today().strftime('%Y%m%d_EXP'))
    parser.add_argument('--settings', default='default_settings.json')
    parser.add_argument('--data_dir', help='Assign a data directory.')
    parser.add_argument('--mem_limit', type=int, help='Limit GPU memory (in MiB) allocated by tf.')
    main(parser.parse_args())