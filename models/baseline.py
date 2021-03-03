from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def baseline_model(input_shape, classes=2):
    """Baseline model for PD classification from https://doi.org/10.1007/s00330-019-06327-0."""
    inp = Input(input_shape, dtype='float')
    x = Conv2D(32, 3)(inp)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPool2D(2)(x)
    x = Dropout(.5)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPool2D(2)(x)
    x = Dropout(.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.5)(x)
    out = Dense(classes, activation='softmax')(x)
    return Model(inp, out, name='baseline_model')