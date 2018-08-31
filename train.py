import os
import datetime
from PIL import Image
from keras.optimizers import *
from keras.models import Model
from keras.models import model_from_json
from keras.layers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils import to_categorical
from sklearn.utils import class_weight
from generator import *
from utils import init_globals, plot_history
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
epochs = 100
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
output_path = 'output/'
fashion_dataset_path = '../Data/fashion_data/'
dataset_path = '../Data/dataset_df_all'
dataset_train_path = os.path.join(dataset_path, 'train')
dataset_val_path = os.path.join(dataset_path, 'validation')
dataset_test_path = os.path.join(dataset_path, 'test')
btl_path = 'E:\\ML\\bottleneck_df_all'
btl_train_path = os.path.join(btl_path, 'train')
btl_val_path = os.path.join(btl_path, 'validation')
btl_test_path = os.path.join(btl_path, 'test')
#
# def get_optimizer(optimizer='Adagrad', lr=None, decay=0.0, momentum=0.0):
#
#     if optimizer == 'SGD':
#         if lr is None:
#             lr = 0.01
#         optimizer_mod = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
#
#     elif optimizer == 'RMSprop':
#         if lr is None:
#             lr = 0.001
#         optimizer_mod = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=decay)
#
#     elif optimizer == 'Adagrad':
#         if lr is None:
#             lr = 0.01
#         optimizer_mod = keras.optimizers.Adagrad(lr=lr, epsilon=1e-08, decay=decay)
#
#     elif optimizer == 'Adadelta':
#         if lr is None:
#             lr = 1.0
#         optimizer_mod = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
#
#     elif optimizer == 'Adam':
#         if lr is None:
#             lr = 0.001
#         optimizer_mod = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
#     elif optimizer == 'Adamax':
#         if lr is None:
#             lr = 0.002
#         optimizer_mod = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
#     elif optimizer == 'Nadam':
#         if lr is None:
#             lr = 0.002
#         optimizer_mod = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#     else:
#         logging.error('Unknown optimizer {}'.format(optimizer))
#         exit(1)
#     logging.debug('lr {}'.format(lr))
#     logging.debug('optimizer_mod {}'.format(optimizer_mod))
#     return optimizer_mod, lr

def create_model(is_input_bottleneck, input_shape):

    if is_input_bottleneck is True:
        model_inputs = Input(shape=(input_shape), name='input_cls_attr')
        common_inputs = model_inputs
    # Predict
    else:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        # shape=(?, 7, 7, 512)
        model_inputs = base_model.input
        common_inputs = base_model.output

    ## Classes
    x = Flatten(name='flatten_cls')(common_inputs)
    x = Dense(256, activation='tanh', name='dense_1_cls')(x)
    x = Dropout(0.5, name='drop_1_cls')(x)
    x = Dense(1024, activation='tanh', name='dense_2_cls')(x)
    x = Dropout(0.5, name='drop_2_cls')(x)
    predictions_class = Dense(len(class_names), activation='softmax', name='predictions_class')(x)

    # Attributes
    x = Flatten(name='flatten_attr')(common_inputs)
    x = Dense(256, activation='tanh', name='dense_1_attr')(x)
    x = Dropout(0.5, name='drop_1_attr')(x)
    x = Dense(2048, activation='tanh', name='dense_2_attr')(x)
    x = Dropout(0.5, name='drop_2_attr')(x)
    predictions_attr = Dense(len(attr_names), activation='sigmoid', name='predictions_attr')(x)

    ## Create Model
    # model = Model(inputs=model_inputs, outputs=[predictions_class, predictions_iou])
    # model = Model(inputs=model_inputs, outputs=predictions_iou)
    model = Model(inputs=model_inputs, outputs=[predictions_class, predictions_attr])
    if is_input_bottleneck is False:
        for layer in model.layers[:19]:
            layer.trainable = False
    logging.info('summary:{}'.format(model.summary()))
    return model

def train_model(batch_size):
    train_labels_class = []
    train_labels_attr = []
    with open(os.path.join(btl_path, 'btl_train.txt')) as f:
        for name in f:
            train_labels_class.append(class_names.index(name.split()[1]))
            if len(name.split()) == 3:
                for i in list(map(int, name.split()[2].split('-'))):
                    train_labels_attr.append(i)
    train_labels_class = np.array(train_labels_class)
    train_labels_attr = np.array(train_labels_attr)

    with open(os.path.join(btl_path, 'btl_validation.txt')) as f:
        for i, l in enumerate(f):
            pass
    val_data_len = i + 1

    ## Build network
    ## Register Callbacks
    log_path = os.path.join(output_path, 'model_train.csv')
    csv_log = CSVLogger(log_path , separator=';', append=False)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='min')
    #filepath = "output/best-weights-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
    filepath = os.path.join(output_path, 'best-weights-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [csv_log, early_stopping, checkpoint]
    # class_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels_class), train_labels_class)
    # attr_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels_attr), train_labels_attr)
    input_shape = (img_width, img_height, img_channel)
    input_shape = (7, 7, 512)
    # model = create_model(is_input_bottleneck=True, is_load_weights=False, input_shape, optimizer, learn_rate, decay, momentum, activation, dropout_rate)
    model = create_model(True, input_shape)
    with open(os.path.join(output_path, 'bottleneck_fc_model.json'), 'w') as f:
        f.write(model.to_json())
    ## Compile
    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss={'predictions_class': 'categorical_crossentropy', 'predictions_attr': 'binary_crossentropy'},
                  metrics=['accuracy'])
    # train_gen = np_arrays_reader(os.path.join(btl_path, 'btl_train_npz.txt'))
    # val_gen = np_arrays_reader(os.path.join(btl_path, 'btl_validation_npz.txt'))
    # val_data = []
    # val_lbls = []
    # for btl_name in sorted(glob.glob(btl_val_path + '/*.npz')):
    #     temp = np.load(open(btl_name, 'rb'))
    #     val_data.append(temp['btl'])
    #     val_lbls.append(temp['iou'])    # #
    # val_data = np.array(val_data)
    # val_lbls = np.array(val_lbls)
    t_begin = datetime.datetime.now()
    with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_train_npz.txt'), 100) as train_gen:
        with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_validation_npz.txt'), 50) as val_gen:
            model.fit_generator(train_gen,
                                    steps_per_epoch=len(train_labels_class) // batch_size,
                                    epochs=epochs,
                                    validation_data=val_gen,
                                    validation_steps=val_data_len // batch_size,
                                    # class_weight=[class_weight_val, attr_weight_val],
                                    # use_multiprocessing=True,
                                    callbacks=callbacks_list)
									# predictions_attr_acc predictions_attr_loss val_predictions_attr_acc val_predictions_attr_loss
									# predictions_class_acc predictions_class_loss val_predictions_class_acc val_predictions_class_loss
    print(datetime.datetime.now())
    print('total_time: {}'.format(str(datetime.datetime.now() - t_begin)))
    # TODO: These are not the best weights
    model.save_weights(os.path.join(output_path, 'bottleneck_fc_model.h5'))
    plot_history(log_path)
### MAIN ###
if __name__ == '__main__':
    global class_names, input_shape, attr_names
    class_names, input_shape, attr_names = init_globals()
    train_model(256)