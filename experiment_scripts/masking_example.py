import tensorflow as tf
from util import masking
import numpy as np

def main():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range and reshape
    x_train = np.reshape(x_train.astype("float32") / 255, (-1,784))
    x_test =  np.reshape(x_test.astype("float32") / 255, (-1,784))

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Construct simple MLP
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(300,activation="relu",name='dense_300'),
            tf.keras.layers.Dense(100,activation="relu",name='dense_100'),
            tf.keras.layers.Dense(10,activation="softmax",name='dense_10'),
        ]
    )

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Fit model
    model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=10)

    print('Accuracy')
    print(model.evaluate(x_test,y_test))

    # Create the module model
    module_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(300,activation="relu",name='dense_300',trainable=False),
            masking.GumbelSigmoidMask(alpha=1e-4,name='mask_300'),
            tf.keras.layers.Dense(100,activation="relu",name='dense_100',trainable=False),
            masking.GumbelSigmoidMask(alpha=1e-4,name='mask_100'),
            tf.keras.layers.Dense(2,activation="softmax",name='head'),
        ]
    )

    # Build and compile model
    module_model.build((None,784))
    module_model.compile(loss="categorical_crossentropy", optimizer="adam")

    # Insert weights from previous model
    module_model.get_layer('dense_300').set_weights(model.get_layer('dense_300').get_weights())
    module_model.get_layer('dense_100').set_weights(model.get_layer('dense_100').get_weights())

    # Prepare the binary data
    binary_y_train = np.column_stack((np.sum(y_train[:,[0]],axis=1),
                                    np.sum(y_train[:,list(set(range(10))-set([0]))],axis=1)))
    binary_y_test = np.column_stack((np.sum(y_test[:,[0]],axis=1),
                                    np.sum(y_test[:,list(set(range(10))-set([0]))],axis=1)))

    # And class weights
    class_0_weight = (1 / np.sum(binary_y_train[:,0])) * (len(binary_y_train) / 2.0)
    class_1_weight = (1 / np.sum(binary_y_train[:,1])) * (len(binary_y_train) / 2.0)

    # Fit the module model
    module_model.fit(x_train,binary_y_train,validation_data=(x_test,binary_y_test),epochs=10,
                class_weight={0:class_0_weight,1:class_1_weight})

    print('Accuracy')
    print(module_model.evaluate(x_test,binary_y_test))

    print('Number of units kept in module')
    masks = [v.weights[0].numpy() for v in module_model.layers if 'mask' in v.name]
    m = np.concatenate([v.reshape(-1) for v in masks])
    print(len(m[m>0.0]))


if __name__=='__main__':
    main()