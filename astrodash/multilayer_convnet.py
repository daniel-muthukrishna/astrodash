from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input


def build_model(N, ntypes):
    inputs = Input(shape=((N,1)))

    hidden = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    hidden = Conv1D(32, kernel_size=3, activation='relu')(hidden)
    hidden = Flatten()(hidden)
    outputs = Dense(ntypes, activation='softmax')(hidden)

    model = Model(inputs, outputs)

    return model

