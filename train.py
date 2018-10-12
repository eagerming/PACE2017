import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Reshape, merge, Concatenate, Flatten, Dropout, concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
import dataset


def init_normal(shape, name=None):
    return initializers.normal(shape)


def get_Model(num_users, num_items, latent_dim, user_con_len, item_con_len, layers=[20, 10, 5], regs=[0, 0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                               embeddings_initializer='uniform', W_regularizer=l2(regs[0]), input_length=1)
    item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                               embeddings_initializer='uniform', W_regularizer=l2(regs[1]), input_length=1)

    user_latent = Flatten()(user_embedding(user_input))
    item_latent = Flatten()(item_embedding(item_input))

    vector = concatenate([user_latent, item_latent])

    for i in range(len(layers)):
        hidden = Dense(layers[i], activation='relu', init='lecun_uniform', name='ui_hidden_' + str(i))
        vector = hidden(vector)

    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)

    user_context = Dense(user_con_len, activation='sigmoid', init='lecun_uniform', name='user_context')(user_latent)
    item_context = Dense(item_con_len, activation='sigmoid', init='lecun_uniform', name='item_context')(item_latent)

    model = Model(input=[user_input, item_input], output=[prediction, user_context, item_context])
    return model


model = get_Model(100000, 100000, 10, 37002, 12223)
config = model.get_config()
weights = model.get_weights()


def get_train_instances(train_data):
    while 1:
        user_input = train_data['user_input']
        item_input = train_data['item_input']
        ui_label = train_data['ui_label']
        u_context = train_data['u_context']
        s_context = train_data['s_context']
        for i in range(len(u_context)):
            u = []
            it = []
            p = []
            u.append(user_input[i])
            it.append(item_input[i])
            p.append(ui_label[i])
            x = {'user_input': np.array(u), 'item_input': np.array(it)}
            y = {'prediction': np.array(p), 'user_context': np.array(u_context[i]).reshape((1, 37002)),
                 'item_context': np.array(s_context[i]).reshape((1, 12223))}
            yield (x, y)


train = None
with open('data/traindata_small.pkl', 'rb') as f:
    train = pickle.load(f)


user_input = train['user']
item_input = train['spot']
ui_label = train['label']
data = dataset.Dataset('_small')
data.generateContextLabels()
contexts = data.context_data
u_context, s_context = contexts['user_context'], contexts['spot_context']
train_data = {}
train_data['user_input'] = user_input
train_data['item_input'] = item_input
train_data['ui_label'] = ui_label
train_data['u_context'] = u_context
train_data['s_context'] = s_context

if __name__ == '__main__':
    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    learner = "Adam"
    learning_rate = 0.0001
    epochs = 100
    batch_size = 1024
    verbose = 1
    losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']

    num_users, num_items = len(user_input), len(item_input)
    num_user_context = len(u_context[0])
    num_item_context = len(s_context[0])

    print('Build model')


    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accs = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accs.append(logs.get('acc'))


    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)

    history = LossHistory()
    model = get_Model(num_users, num_items, 10, 37002, 12223, layers, reg_layers)

    model.compile(optimizer=Adam(lr=learning_rate), loss=losses, metrics=['accuracy'])

    print('Start Training')

    for epoch in range(epochs):
        t1 = time()
        hist = model.fit_generator(get_train_instances(train_data), samples_per_epoch=batch_size, nb_epoch=10,
                                   verbose=1, callbacks=[history, board])
        t2 = time()
        print(epoch, t2 - t1)