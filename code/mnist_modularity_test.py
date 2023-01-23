import mask_utils
import numpy as np
import tensorflow as tf
import pandas as pd
import gc
import pickle

def main():
    """
    - import mnist
    - for network width
        - for random init seed
            - create network
            - train network
            - for class in classes
                - add mask layers
                - train module
                - save: accuracy/loss of network, accuracy/loss of module, 
                        module mask for each layer,
    """
    print(tf.config.list_physical_devices())

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train_identities,y_test_identities = y_train,y_test

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    experimental_stats = pd.DataFrame([],columns=['width','iteration','label','alpha','parameters',
                                                  'model_loss','model_accuracy','model_auc',
                                                  'model_precision','model_recall','model_crossentropy','gumbel_model_loss',
                                                  'gumbel_model_accuracy','gumbel_model_auc','gumbel_model_precision',
                                                  'gumbel_model_recall','gumbel_crossentropy','units_kept','sum_of_sigmoids'])

    METRICS = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.CategoricalCrossentropy()
        ]

    for width in [2,4,6,8]:
        for iteration in range(1):
            print(width,iteration)

            model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.ZeroPadding2D(2),
                tf.keras.layers.Conv2D(2**width, kernel_size=(5, 5), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
                tf.keras.layers.ZeroPadding2D(2),
                tf.keras.layers.Conv2D(2**width, kernel_size=(5, 5), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
                tf.keras.layers.ZeroPadding2D(2),
                tf.keras.layers.Conv2D(2**width, kernel_size=(5, 5), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2**(width+1),activation="relu"),
                tf.keras.layers.Dense(2**(width+1),activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
            )

            model.compile(loss="categorical_crossentropy", 
                        optimizer="adam", metrics=METRICS)

            model.fit(x_train, y_train, validation_data=(x_test,y_test), 
                    epochs=100, batch_size=256,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    min_delta=0,
                                    patience=5,
                                    restore_best_weights=True,
                                )]
                        )

            num_trainable_parameters = sum([np.prod(v.shape) for v in model.trainable_variables])
            print (num_trainable_parameters)

            evaluation_metrics = [model.evaluate(x_test[y_test[:,i]==1],
                                            y_test[y_test[:,i]==1], batch_size=256) for i in range(10)]
            model_loss = [x[0] for x in evaluation_metrics]
            model_accuracy = [x[1] for x in evaluation_metrics]
            model_auc = [x[2] for x in evaluation_metrics]
            model_precision = [x[3] for x in evaluation_metrics]
            model_recall = [x[4] for x in evaluation_metrics]
            model_crossentropy = [x[5] for x in evaluation_metrics]

            model.save('../data/models/model_width_'+str(width)+'_iteration_'+str(iteration))

            for label in [0,1]:
                binary_y_train = np.column_stack((np.sum(y_train[:,[label]],axis=1),
                                        np.sum(y_train[:,list(set(range(10))-set([label]))],axis=1)))
                binary_y_test = np.column_stack((np.sum(y_test[:,[label]],axis=1),
                                        np.sum(y_test[:,list(set(range(10))-set([label]))],axis=1)))

                class_0_weight = (1 / np.sum(binary_y_train[:,0])) * (len(binary_y_train) / 2.0)
                class_1_weight = (1 / np.sum(binary_y_train[:,1])) * (len(binary_y_train) / 2.0)
                print('class 0',np.sum(binary_y_train[:,0]),class_0_weight)
                print('class 1',np.sum(binary_y_train[:,1]),class_1_weight)

                for alpha in [1e-7,1e-6,1e-5,1e-4,1e-3]:

                    model = tf.keras.models.load_model('../data/models/model_width_'+str(width)+'_iteration_'+str(iteration))

                    gumbel_model = mask_utils.add_masks(model,mask_type='gumbel',
                                                        layer_kwargs={'alpha':alpha,'regularizer':'SumOfSigmoidsRegularizer',
                                                                      'num_mask_samples':8})

                    gumbel_model.build((None,28,28,1))
                    gumbel_model.compile(loss="categorical_crossentropy", 
                                        optimizer="adam", metrics=METRICS)
                    gumbel_model.fit(x_train,binary_y_train, batch_size=256,
                                    validation_data=(x_test,binary_y_test),epochs=1000,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            mode='min',
                                            min_delta=0,
                                            patience=50,
                                            restore_best_weights=True,
                                            verbose=1,
                                        )],
                                    class_weight={0:class_0_weight,1:class_1_weight})

                    gumbel_loss,gumbel_accuracy,gumbel_auc,gumbel_precision,gumbel_recall,gumbel_crossentropy = gumbel_model.evaluate(x_test,
                                                                                            binary_y_test, batch_size=256)

                    masks = [v.weights[0].numpy() for v in gumbel_model.layers if 'mask' in v.name]
                    m = np.concatenate([v.reshape(-1) for v in masks])

                    experimental_stats.loc[len(experimental_stats)] = [width,iteration,label,
                                                                       alpha,num_trainable_parameters,
                                                                       model_loss[label],
                                                                       model_accuracy[label],
                                                                       model_auc[label],
                                                                       model_precision[label],
                                                                       model_recall[label],
                                                                       model_crossentropy[label],
                                                                       gumbel_loss,
                                                                       gumbel_accuracy,
                                                                       gumbel_auc,
                                                                       gumbel_precision,
                                                                       gumbel_recall,
                                                                       gumbel_crossentropy,
                                                                       len(m[m>0.0]),
                                                                       np.sum(tf.math.sigmoid(m))]

                    experimental_stats.to_csv('../data/experimental_stats.csv')
                    
                    with open('../data/masks/gumbel_model_width_'+str(width)+
                                      '_iteration_'+str(iteration)+'_label_'+str(label)+
                                      '_alpha_'+str(alpha)+'.pkl', 'wb') as f:
                        pickle.dump(masks,f)

                    del model
                    del gumbel_model
                    gc.collect()

if __name__ == "__main__":
    main()
