import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hybrid_model(input_shape,output_shape,lstm_units,dense_units,key_dim,value_dim,num_heads):

  ''''Modelo con modulos de atencion'''

  inputs=Input((input_shape,1))

  lstm_output=LSTM(lstm_units)(inputs)

  attention_output=MultiHeadAttention(
      num_heads,key_dim,value_dim
  )(inputs,inputs,inputs)

  x=Flatten()(lstm_output)
  y=Flatten()(attention_output)
  z=Flatten()(inputs)

  output_flatten=concatenate([x,y,z])

  h=Dense(dense_units)(output_flatten)

  h=Dense(output_shape)(h)

  output=tf.expand_dims(h,axis=-1)

  model=Model(inputs,output)

  #model.summary()

  return model

def windows_state(series, input_shape=30, output_shape=1, shift=1, batch_size=32):

    '''windows para todos los input y output'''
    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(input_shape + output_shape, shift=shift, drop_remainder=True)

    ds = ds.flat_map(lambda window: window.batch(input_shape + output_shape))

    ds = ds.map(lambda window: (window[:input_shape], window[input_shape:]))

    # ds=ds.shuffle(234)

    ds = ds.batch(batch_size).prefetch(1)

    return ds


def windows_prediction(series, input_shape=30, shift=10):

    '''Sirve para crear datos que predigan'''
    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(input_shape, shift=shift, drop_remainder=True)

    ds = ds.flat_map(lambda window: window.batch(input_shape))

    # ds=ds.shuffle(234)

    ds = ds.batch(1).prefetch(1)

    return ds


def fit_model(input_shape, output_shape, lstm_units, dense_units, lr, epochs):

    '''Es el nucelo donde se junta todo'''
    global train
    global test

    key_dim = 200
    value_dim = 200
    num_heads = 1
    batch_size = 24

    model = hybrid_model(input_shape, output_shape, lstm_units, dense_units, key_dim, value_dim, num_heads)

    model.compile(loss=MeanAbsoluteError(), optimizer=Adam(lr), metrics=['mse'])

    train_ds = windows_state(train, input_shape=input_shape, output_shape=output_shape, shift=1, batch_size=batch_size)

    test_ds = windows_state(test, input_shape=input_shape, output_shape=output_shape, shift=1, batch_size=batch_size)

    with tf.device('/GPU:0'):
        model.fit(train_ds, epochs=epochs, validation_data=test_ds,
                  callbacks=ModelCheckpoint('lstm_attention.h5', save_best_only=True))

    model = load_model('lstm_attention.h5')

    scores = model.evaluate(test_ds)

    return -scores[1]
  
  
  minmax=MinMaxScaler()

'''Optional'''
open=minmax.fit_transform(open)

min=minmax.data_min_

max=minmax.data_max_

#SPLIT

train_size=int(len(open)*8/10)

train=open[:train_size]

test=open[train_size:]

print(min)
print(max)

def create_predictions(model, test, input_shape, output_shape):
    global min

    global max

    test_ds = windows_prediction(test, input_shape=input_shape, shift=output_shape)

    count = 0

    for i in test_ds:
        count += 1

    count -= 1

    r = len(test) - (input_shape + output_shape * count)

    print(r)

    pred_data = model.predict(test_ds)

    shape = pred_data.shape

    pred_data = np.reshape(pred_data, (shape[0] * shape[1], 1))
    pred_data = pred_data[:-output_shape, :]
    real_data = test[input_shape:-r]

    # Inverse Scaling
    real_data = real_data * (max - min) + min
    pred_data = pred_data * (max - min) + min

    print(pred_data.shape)
    print(real_data.shape)

    figure = plt.figure()
    figure.set_size_inches(10, 10)
    plt.plot(real_data, label='real_data')
    plt.plot(pred_data, label='pred_data')
    plt.legend()

    print(f'MEAN ABSOLUTE ERROR:{mean_absolute_error(real_data, pred_data)}')
