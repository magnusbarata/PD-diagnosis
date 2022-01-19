from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv3D, MaxPool3D, GlobalAvgPool3D
from tensorflow.keras.models import Model

def baseline(input_shape, classes=2):
    """Baseline model for PD classification from https://doi.org/10.1007/s00330-019-06327-0."""
    if len(input_shape) == 4:
        conv_layer = Conv3D
        pool_layer = MaxPool3D
        model_name = 'baseline_model_3d'
    else:
        conv_layer = Conv2D
        pool_layer = MaxPool2D
        model_name = 'baseline_model'
    
    inp = Input(input_shape, dtype='float')
    x = conv_layer(32, 3)(inp)
    x = conv_layer(32, 3, activation='relu')(x)
    x = pool_layer(2)(x)
    x = Dropout(.5)(x)
    x = conv_layer(64, 3, activation='relu', padding='same')(x)
    x = conv_layer(64, 3, activation='relu')(x)
    x = pool_layer(2)(x)
    x = Dropout(.5)(x)
    x = GlobalAvgPool3D()(x) if len(input_shape) == 4 else Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.5)(x)
    out = Dense(classes, activation='softmax')(x)
    return Model(inp, out, name=model_name)
