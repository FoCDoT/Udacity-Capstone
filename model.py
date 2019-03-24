import json

from keras.models import Model, model_from_json
from keras.layers import Dense, LSTM, Input, LeakyReLU, GlobalAveragePooling1D, concatenate, Dropout, Flatten, CuDNNLSTM
from keras.layers import Bidirectional, BatchNormalization, Embedding
from keras.initializers import Constant
from keras.regularizers import l2


def save_model(model, savepath, weights_name=None, model_name='model.json'):
    """
    serialize model and weights
    :param model: Keras  model
    :param savepath: path to save model and weight
    :param weights_name: weights file name
    :param model_name: model file name
    :return:
    """
    model_json = model.to_json()

    with open(savepath + model_name, 'w') as json_file:
        json_file.write(model_json)

    # serialize weights
    if weights_name:
        model.save_weights(savepath + weights_name)
    print('Saved model')


def restore_model(savepath, weights_name=None, model_name='model.json'):
    """
        load model and weights
        :param savepath: path to save model and weight
        :param weights_name: weights file name
        :param model_name: model file name
        :return:
    """
    with open(savepath + model_name) as json_file:
        model = model_from_json(json_file.read())
    if weights_name:
        model.load_weights(savepath + weights_name)
    print('Restored model')
    return model


def create_model(args):
    inp = Input((args['max_seq_len'],))
    qset = Input((1,))

    x = Embedding(args['nb_words'], args['embed_dim'], input_length=args['max_seq_len'])(inp)
    x = Bidirectional(CuDNNLSTM(250, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = concatenate([x, qset])  # functional interface for concat
    x = BatchNormalization()(x)
    x = Dense(1, kernel_initializer='normal', bias_initializer=Constant(1.5),
              kernel_regularizer=l2())(x)

    model = Model(inputs=[inp, qset], outputs=x)
    return  model


if __name__ == '__main__':
    with open('config.json') as f:
        args = json.load(f)

    model = create_model(args)
    model.summary()
