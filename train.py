from math import exp

import matplotlib.pyplot as plt
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import keras.backend as K
from sklearn.model_selection import train_test_split


def val_split(word_seq, X_set, y, split_ratio=0.8):
    """
    :param word_seq: keras tokenized text array
    :param set: question set
    :param y: targets
    :return: tuple of (x, y) arrays | x = (text,  set) stratified on set
    """
    X_text_train, X_text_val, X_set_train, X_set_val, y_train, y_val = train_test_split(word_seq, X_set, y,
                                                                                           test_size=1-split_ratio,
                                                                                           random_state=42,
                                                                                           stratify=X_set)
    train = [[X_text_train, X_set_train], y_train]
    test = [[X_text_val, X_set_val], y_val]

    return train, test


def plot_model_history(model_history, savepath):
    # summarize history for loss
    ax1 = plt.subplot(211)
    plt.plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    plt.plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    plt.title('Model History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(range(1, len(model_history.history['lr'])+1), model_history.history['lr'])
    plt.ylabel('Lr')
    plt.xlabel('Epoch')
    plt.legend(['lr'], loc='best')

    plt.savefig(savepath + 'model_history.png')
    plt.show()


def accuracy(y_true, y_pred):
    y_pred = K.round(y_pred)
    acc = K.mean(K.equal(y_true, y_pred))
    return acc


def fit_model(model, train_data, val_data, args):
    """
    :param model: Keras  model
    :param train_data: tuple of (x, y) array
    :param val_data: tuple of (x, y) array
    :param args:  dict containing hyperparameters
    note: x above is tuple/list of text question set pair
    :return: trained model object and history

    data format:
    X_text_train, X_set_train  = train_data[0]  # features
    y_train = train_data[1] # target
    """
    #  compile the model
    # adam = Adam(args['lr'], decay=args['decay'], epsilon=1e-08)
    rmsp = RMSprop(args['lr'], clipvalue=10)
    model.compile(loss='mean_squared_error', optimizer=rmsp,
                  metrics=[accuracy])

    # define callbacks
    tb_log_dir = args['save_folder'] + 'log'
    cp_dir= args['save_folder'] + 'model.weights.best.hdf5'

    tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=1,
                             write_graph=True,
                             write_grads=True,
                             batch_size=args['batch_size'])

    model_cp = ModelCheckpoint(filepath=cp_dir,
                               verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=1e-6, verbose=1)

    def exp_decay(epoch, lr):
        if (epoch + 1) % 3 == 0 and epoch <= 10: # decrease once ever 3 epoch til 10
            k = 0.1
            lrate = lr * exp(-k * (epoch+1))
            return lrate
        if (epoch + 1) % 5 == 0:
            k = 0.1
            lrate = lr * exp(-k * (epoch + 1))
            return lrate
        return lr

    exp_lr = LearningRateScheduler(exp_decay)

    callbacks = [model_cp, tb_callback, reduce_lr, exp_lr]

    # train  the model
    model_hist = model.fit(train_data[0], train_data[1],
                           batch_size=args['batch_size'], epochs=args['epochs'],
                           callbacks=callbacks,
                           validation_data=(val_data[0], val_data[1]),
                           shuffle=True, verbose=2)

    return model, model_hist

