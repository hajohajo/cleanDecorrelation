import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from disco_tf import distance_corr
import numpy as np
import neural_structured_learning as nsl

class DiscoLoss(Loss):
    def __init__(self, factor=15.0, reduction=losses_utils.ReductionV2.AUTO, name="DiscoLoss"):
        super().__init__(reduction=reduction, name=name)
        self.factor = factor

    def call(self, y_true, y_pred):
        # Split given labels to the target and the mT value needed for decorrelation
        y_pred = tf.convert_to_tensor(y_pred)
        sample_weights = tf.cast(tf.reshape(y_true[:, 2], (-1, 1)), y_pred.dtype)
        mt = tf.cast(tf.reshape(y_true[:, 1], (-1, 1)), y_pred.dtype)
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)

        dcPred = tf.reshape(y_pred, [tf.size(y_pred)])
        dcMt = tf.reshape(mt, [tf.size(mt)])
        weights = tf.cast(tf.reshape(sample_weights, [tf.size(sample_weights)]), y_pred.dtype)

        # The loss
        custom_loss = tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0) + self.factor * distance_corr(dcMt, dcPred, normedweight=weights, power=1)
        return custom_loss

#AUC that can be used as a metric when using the discoLoss with 3D target
def get_custom_auc():
    auc = tf.metrics.AUC()
    # @tf.function
    def custom_auc(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)
        auc.update_state(y_true, y_pred)
        return auc.result()

    custom_auc.__name__ = "custom_auc"
    return custom_auc

class InputSanitizerLayer(tf.keras.layers.Layer):
    def __init__(self, minValues, maxValues, **kwargs):
        self.minValues = minValues
        self.maxValues = maxValues
        self.tensorMin = tf.convert_to_tensor(np.reshape(self.minValues, (1, self.minValues.shape[-1])), dtype='float32')
        self.tensorMax = tf.convert_to_tensor(np.reshape(self.maxValues, (1, self.maxValues.shape[-1])), dtype='float32')
        super(InputSanitizerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InputSanitizerLayer, self).build(input_shape)

    def call(self, input):
        values = tf.math.multiply(tf.math.divide((tf.math.maximum(tf.math.minimum(input, self.tensorMax), self.tensorMin)
                                                  - self.tensorMin), (self.tensorMax - self.tensorMin)), 2) - 1.0
        return values

    def get_config(self):
        return {'minValues': self.minValues, 'maxValues': self.maxValues}



def get_disco_classifier(n_inputs, min_values, max_values):
    _neurons = 128
    _activation = 'relu'
    _regularization = tf.keras.regularizers.l2(1e-4)
    _initializer = 'lecun_normal'
    _optimizer = tf.keras.optimizers.Adam(lr=3e-4, amsgrad=True)
    _optimizer2 = tf.keras.optimizers.Adam(lr=3e-4, amsgrad=True)
    _droprate = 0.2
    _layers = 1

    input_layer = tf.keras.layers.Input(n_inputs, name='feature')
    x = InputSanitizerLayer(min_values, max_values)(input_layer)
    dense_1 = tf.keras.layers.Dense(_neurons, activation=_activation, kernel_initializer=_initializer, activity_regularizer=_regularization)(x)
    batch_1 = tf.keras.layers.BatchNormalization()(dense_1)
    drop = tf.keras.layers.Dropout(_droprate)(batch_1)
    for i in range(_layers):
        dense = tf.keras.layers.Dense(_neurons, activation=_activation, kernel_initializer=_initializer,
                                        activity_regularizer=_regularization)(drop)
        batch = tf.keras.layers.BatchNormalization()(dense)
        drop = tf.keras.layers.Dropout(_droprate)(batch)
    dense = tf.keras.layers.Dense(_neurons, activation=_activation, kernel_initializer=_initializer,
                                    activity_regularizer=_regularization)(drop)
    batch = tf.keras.layers.BatchNormalization()(dense)

    # dense_2 = tf.keras.layers.Dense(_neurons, activation=_activation, kernel_initializer=_initializer, activity_regularizer=_regularization)(drop_1)
    # batch_2 = tf.keras.layers.BatchNormalization()(dense_2)
    # drop_2 = tf.keras.layers.Dropout(_droprate)(batch_2)
    # dense_3 = tf.keras.layers.Dense(_neurons, activation=_activation, kernel_initializer=_initializer, activity_regularizer=_regularization)(drop_2)
    # batch_3 = tf.keras.layers.BatchNormalization()(dense_3)
    # drop_3 = tf.keras.layers.Dropout(_droprate)(batch_3)
    # dense_4 = tf.keras.layers.Dense(_neurons, activation=_activation, kernel_initializer=_initializer, activity_regularizer=_regularization)(drop_3)
    # batch_4 = tf.keras.layers.BatchNormalization()(dense_4)
    # drop_4 = tf.keras.layers.Dropout(_droprate)(batch_4)

    # shape = tf.keras.layers.Reshape((1, 8))(input_layer)
    # conv1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(shape)
    # drop1 = tf.keras.layers.Dropout(_droprate)(conv1)
    # conv2 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop1)
    # drop2 = tf.keras.layers.Dropout(_droprate)(conv2)
    # conv3 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop2)
    # drop3 = tf.keras.layers.Dropout(_droprate)(conv3)
    # conv4 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop3)
    # drop4 = tf.keras.layers.Dropout(_droprate)(conv4)
    # conv5 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop4)
    # drop5 = tf.keras.layers.Dropout(_droprate)(conv5)
    # conv6 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop5)
    # drop6 = tf.keras.layers.Dropout(_droprate)(conv6)
    # conv7 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop6)
    # drop7 = tf.keras.layers.Dropout(_droprate)(conv7)
    # conv8 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(drop7)
    # flat = tf.keras.layers.Flatten()(conv8)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=_initializer)(batch)

    model = tf.keras.Model(input_layer, output_layer)
    model_naive = tf.keras.Model(input_layer, output_layer)
    adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
    # adv_model = nsl.keras.AdversarialRegularization(model_naive, adv_config=adv_config)
    adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)
    adv_model_naive = nsl.keras.AdversarialRegularization(model_naive, adv_config=adv_config)

    # model.compile(optimizer=_optimizer2,
    #               metrics=[get_custom_auc()],
    #               loss=DiscoLoss(10.0)) #(5.00))
    adv_model.compile(optimizer=_optimizer2,
                  metrics=[get_custom_auc()],
                  loss=DiscoLoss(factor=10.0)) #(5.00))
    adv_model_naive.compile(optimizer=_optimizer,
                  metrics=[tf.keras.metrics.AUC()],
                  loss="binary_crossentropy")
    model_naive.compile(optimizer=_optimizer,
                  metrics=[tf.keras.metrics.AUC()],
                  loss="binary_crossentropy")


    return adv_model, adv_model_naive, model_naive

def get_autoencoder(n_inputs):
    _activation = 'relu'
    _regularization = tf.keras.regularizers.l2(1e-5)
    _initializer = 'lecun_normal'
    _optimizer = tf.keras.optimizers.Adam(lr=3e-3, amsgrad=True)
    _droprate = 0.2

    input_layer = tf.keras.layers.Input(n_inputs)
    reshape = tf.keras.layers.Reshape((1, 8))(input_layer)
    conv1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(reshape)
    conv2 = tf.keras.layers.Conv1D(2, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(conv1)
    conv3 = tf.keras.layers.Conv1D(32, kernel_size=1, activation=_activation, kernel_initializer=_initializer)(conv2)
    flat = tf.keras.layers.Flatten()(conv3)
    out = tf.keras.layers.Dense(n_inputs, activation="linear", kernel_initializer=_initializer)(flat)

    model = tf.keras.Model(input_layer, out)
    model.compile(optimizer=_optimizer,
                  loss="mae")
    print(model.summary())
    return model
