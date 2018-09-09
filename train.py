import os
import sys
import datetime
import time
import pickle
from PIL import Image
from keras import backend as K
from keras.optimizers import *
from keras.models import Model
from keras.models import model_from_json, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils import plot_model
from sklearn.utils import class_weight
from generator import *
from utils import init_globals, plot_history
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
output_path = 'output/'
fashion_dataset_path = '../Data/fashion_data/'
# dataset_path = '../Data/dataset_df'
# dataset_train_path = os.path.join(dataset_path, 'train')
# dataset_val_path = os.path.join(dataset_path, 'validation')
# dataset_test_path = os.path.join(dataset_path, 'test')
btl_path = 'E:\\ML\\bottleneck_dfn'
btl_train_path = os.path.join(btl_path, 'train')
btl_val_path = os.path.join(btl_path, 'validation')
btl_test_path = os.path.join(btl_path, 'test')

def create_model(is_input_bottleneck, input_shape, classes, attributes, mode=3):

    if is_input_bottleneck is True:
        model_inputs = Input(shape=(input_shape), name='input_vgg16')
        common_inputs = model_inputs
    # Predict
    else:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        # shape=(?, 7, 7, 512)
        model_inputs = base_model.input
        common_inputs = base_model.output

    input_flatten = Flatten()(common_inputs)

    # Classes
    x = Dense(512, activation='elu', name='dense_1_cls')(input_flatten)
    x = BatchNormalization(name='bn_1_cls')(x)
    # x = Dropout(0.5, name='drop_1_cls')(x)
    # x = Dense(512, activation='elu', name='dense_2_cls')(x)
    # x = BatchNormalization(name='bn_2_cls')(x)
    x = Dense(512, activation='elu', name='dense_2_cls')(x)
    x = BatchNormalization(name='bn_2_cls')(x)
    x = Dense(512, activation='elu', name='dense_3_cls')(x)
    x = BatchNormalization(name='bn_3_cls')(x)
    # x = Dropout(0.5, name='drop_2_cls')(x) 
    predictions_class = Dense(len(classes), activation='softmax', name='predictions_class')(x)

    # Bboxes
    x = Dense(512, activation='elu', name='dense_1_bbox')(input_flatten)
    x = BatchNormalization(name='bn_1_bbox')(x)
    x = Dense(512, activation='elu', name='dense_3_bbox')(x)
    x = BatchNormalization(name='bn_3_bbox')(x)
    predictions_bbox = Dense(4, activation='sigmoid', name='predictions_bbox')(x)

    # Attributes
    x = Dense(512, activation='elu', name='dense_1_attr')(input_flatten)
    x = BatchNormalization(name='bn_1_attr')(x)
    x = Dense(512, activation='elu', name='dense_3_attr')(x)
    # merge_layer = concatenate([head_bbox, head_attr, head_cls])
    x = BatchNormalization(name='bn_3_attr')(x)
    x = Dense(512, activation='elu', name='dense_4_attr')(x)
    x = BatchNormalization(name='bn_4_attr')(x)
    predictions_attr = Dense(len(attributes), activation='sigmoid', name='predictions_attr')(x)
    ## Create Model
    if mode == 'bac':
        model = Model(inputs=model_inputs, outputs=[predictions_bbox, predictions_attr, predictions_class]) 
    if mode == 'c':
        model = Model(inputs=model_inputs, outputs=predictions_class)
        # for layer in model.layers:
        #     if layer.name in ['dense_1_bbox','dense_2_bbox','predictions_bbox','dense_1_cls','dense_2_cls', 'predictions_class']:
        #         layer.trainable = False
        #         logging.info('Not taining layer: {}'.format(layer.name)) 
                
    if is_input_bottleneck is False:
        for layer in model.layers[:15]:
            logging.info('Not taining layer: {}'.format(layer.name)) 
            layer.trainable = False
    # logging.info('summary:{}'.format(model.summary()))
    return model

def custom_loss(y_true, y_pred):
    y_true_present = y_true[..., 0:1]
    y_pred_present = y_pred[..., 0:1]
    loss = (1 - y_true_present) * K.categorical_crossentropy(y_true[..., 1:], y_pred[..., 1:]) + \
           K.binary_crossentropy(y_true_present, y_pred_present)
    return loss

def train_model():                 
    # factor = 1
    # val_steps = factor*60
    train_steps = 340
    epochs = 200
    with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_train_npz95.txt'), ['cls'], maxsize=int(0.15*train_steps), numproc=3) as train_gen:
        # with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_validation_npz5.txt'), ['bbiou', 'attr', 'cls'], maxsize=val_steps) as val_gen:
    # with Parallel_image_read_transformer(os.path.join(fashion_dataset_path, 'train85.txt'), 32, class36, attr200, 10) as train_gen:
    #     with Parallel_image_read_transformer(os.path.join(fashion_dataset_path, 'validation8545.txt'), 32, class36, attr200, 10) as val_gen:
        temp = np.load(os.path.join(btl_path, 'test5.npz'))
        X = np.array(temp['btl'])
        # Yb = np.array(temp['bbiou'][:, :4])
        # Ya = np.array(temp['attr'])
        Yc = np.array(temp['cls'])
        # for y in [str(i*256).zfill(7) for i in range(1,60)]:
        #     temp = np.load(os.path.join(btl_path, 'validation\\btl_validation_256_'+y+'.npz'))
        #     X = np.concatenate([X, temp['btl']])
        #     Yb = np.concatenate([Yb, temp['bbiou'][:, :4]])
        #     Ya = np.concatenate([Ya, temp['attr']])
        #     Yc = np.concatenate([Yc, temp['cls']])
        # np.savez(open(os.path.join(btl_path, 'test5.npz'), 'wb'), btl=X,
        #                  cls=Yc, attr=Ya, bbiou=Yb)
        val_data=(X, Yc)
        # with open(os.path.join(btl_path, 'attr_data_train85.pkl'), 'rb') as f:
        #     train_labels_attr = pickle.load(f)
        with open(os.path.join(btl_path, 'class_data_train85n.pkl'), 'rb') as f:
            train_labels_class = pickle.load(f)
        cls_weight = class_weight.compute_class_weight('balanced', class36, train_labels_class)
        # attr_weight = class_weight.compute_class_weight('balanced', attr200, train_labels_attr)
        ## Register Callbacks
        log_path = os.path.join(output_path, 'model_train.csv')
        csv_log = CSVLogger(log_path , separator=';', append=False)
        filepath = os.path.join(output_path, "best-model-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5")
        # early_stop = EarlyStopping(monitor='val_predictions_attr_acc', patience=5, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # early_stop = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # callbacks_list = [csv_log, checkpoint, early_stop]
        callbacks_list = [csv_log, checkpoint]

        model = create_model(True, (7, 7, 512), class36, attr200, mode='c')
        # model.load_weights('output2/final_weights.hdf5', by_name=True)
        # model.load_weights('output4/best-weights.hdf5', by_name=True)
        # # model.predict(X)
        # model.save('models/model2.h5')
        # return
        # with open(os.path.join(output_path, 'bottleneck_fc_model.json'), 'w') as f:
        #     f.write(model.to_json())
        plot_model(model, to_file=os.path.join(output_path, 'model.png'), show_shapes=True, show_layer_names=False)
        ## Compile
        model.compile(
                      optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True),
                      # optimizer=Adam(lr=1e-3),
                      # optimizer=Adadelta(),
                      # optimizer=RMSprop(lr=1e-4, decay=1e-6),
                      loss={
                            # 'predictions_bbox':'mse',
                            # 'predictions_attr':'binary_crossentropy',
                            'predictions_class':custom_loss,
                            },
                      # loss_weights=[0.3,1.,0.3],
                      metrics=['accuracy'])
        t_begin = datetime.datetime.now()
        ## Fit
        model.fit_generator(train_gen, steps_per_epoch=train_steps,
                                epochs=epochs,
                                validation_data=val_data,
                                # validation_steps=val_steps,
                                # class_weight=[[1.,1.,1.,1.], attr_weight],
                                class_weight=cls_weight,
                                # use_multiprocessing=True,
                                callbacks=callbacks_list)
                                    
    print(datetime.datetime.now())
    print('total_time: {}'.format(str(datetime.datetime.now() - t_begin)))
    print('model saved to: {}'.format(output_path))
    # TODO: These are not the best weights
    model.save(os.path.join(output_path, 'final_model.h5'))
    model.save_weights(os.path.join(output_path, 'final_weights.hdf5'))
    plot_history(output_path)
    
### MAIN ###
if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == 'plot':
            plot_history(output_path)
            exit(0)
    else:
        if os.path.exists(output_path):
            i = 1
            while os.path.exists(output_path):
                output_path = 'output%d/' % i
                i += 1
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    global class_names, input_shape, attr_names, class36, attr200
    class_names, input_shape, attr_names = init_globals()
    class36 = ['None', 'Blazer', 'Top', 'Dress', 'Chinos', 'Jersey', 'Cutoffs', 'Kimono', 'Cardigan', 'Jeggings', 'Button-Down',
               'Romper', 'Skirt', 'Joggers', 'Tee', 'Turtleneck', 'Culottes', 'Coat', 'Henley', 'Jeans', 'Hoodie',
               'Blouse', 'Tank', 'Shorts', 'Bomber', 'Jacket', 'Parka', 'Sweatpants', 'Leggings', 'Flannel',
               'Sweatshorts', 'Jumpsuit', 'Poncho', 'Trunks', 'Sweater', 'Robe']
    attr200 = [730, 365, 513, 495, 836, 596, 822, 254, 884, 142, 212, 883, 837, 892, 380, 353, 196, 546, 335, 162, 441,
               717, 760, 568, 310, 705, 745, 81, 226, 830, 620, 577, 1, 640, 956, 181, 831, 720, 601, 112, 820, 935,
               969, 358, 933, 983, 616, 292, 878, 818, 337, 121, 236, 470, 781, 282, 913, 93, 227, 698, 268, 61, 681,
               713, 239, 839, 722, 204, 457, 823, 695, 993, 0, 881, 817, 571, 565, 770, 751, 692, 593, 825, 574, 50,
               207, 186, 237, 563, 300, 453, 897, 944, 438, 688, 413, 409, 984, 191, 697, 368, 133, 676, 11, 754, 800,
               83, 14, 786, 141, 841, 415, 608, 276, 998, 99, 851, 429, 287, 815, 437, 747, 44, 988, 249, 543, 560, 653,
               843, 208, 899, 321, 115, 887, 699, 15, 764, 48, 749, 852, 811, 862, 392, 937, 87, 986, 129, 336, 689,
               245, 911, 309, 775, 638, 184, 797, 512, 45, 682, 139, 306, 880, 231, 802, 264, 648, 410, 30, 356, 531,
               982, 116, 599, 774, 900, 218, 70, 562, 108, 25, 450, 785, 877, 18, 42, 624, 716, 36, 920, 423, 784, 788,
               538, 325, 958, 480, 20, 38, 931, 666, 561]
    logging.info('bottleneck path: {}'.format( btl_path))
    logging.info('output path: {}'.format(output_path))
    train_model()