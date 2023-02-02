# differentiable-unit-masks-tensorflow
Differentiable unit masks inspired by https://arxiv.org/pdf/2010.02066.pdf

This repository implements differientiable units masks in tensorflow. Differentiable unit masks are binary masks applied to units a neural network. These masks can be applied to units in a pre-trained (full) network and optimized via gradient descent to ''discover'' (sub) networks that perform subtasks of the problem the full network was trained to solve. 

For instance, differentiable unit masks can be applied to networks trained on MNIST to discover subnetworks capable of identifying only one digit. They can also be applied to more complex networks trained on more complex tasks to find simpler networks. For instance, by applying masks to networks trained on ImageNet, we can find subnetworks responsible for detecting specific shapes or patterns.

## Setup
To set up your environment and conduct experiments run
```
conda env create --file environment.yml
conda activate dum
conda develop src
```

## Applying masks
Here, we'll run through a simple example for training differentiable weight masks on MNIST. First, we'll train a simple 2-layer MLP model on the full MNIST dataset to classify digits into one of 10 classes. Then, we'll apply differentiable unit mask layers to the model after each of the two layers. The weights of the original model will be frozen, but we'll optimize the mask weights to keep only the units necessary for detecting zeros in the input.

First, load and preprocess the MNIST data.

```
 # Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range and reshape
x_train = np.reshape(x_train.astype("float32") / 255, (-1,784))
x_test =  np.reshape(x_test.astype("float32") / 255, (-1,784))

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

Then, construct and train a simple 2-layer MLP model to classify the inputs into the ten classes.

```
# Construct simple MLP
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(300,activation="relu",name='dense_300'),
        tf.keras.layers.Dense(100,activation="relu",name='dense_300'),
        tf.keras.layers.Dense(10,activation="softmax",name='dense_10'),
    ]
)

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Fit model
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=10)
```

The model here achieves nearly 99% on the validation set. Next we'll apply the `GumbelSigmoidMask` after each of the hidden layers in the network. These mask layers will be trainable while the weights of the previous layers will be frozen. We also add a new classification head on the top of the model with only two outputs for the binary classification problem we plan on solving.

```
module_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(300,activation="relu",name='dense_300',trainable=False),
        masking.GumbelSigmoidMask(alpha=1e-3,name='mask_300'),
        tf.keras.layers.Dense(100,activation="relu",name='dense_100',trainable=False),
        masking.GumbelSigmoidMask(alpha=1e-3,name='mask_100'),
        tf.keras.layers.Dense(2,activation="softmax",name='head'),
    ]
)
```

In addition to add the masks, we must choose an `alpha` hyperparameter value for each layer. The hyperparameter `alpha` determines how heavily we penalize kept units in the module. The more we increase `alpha` the smaller our module will get. As suggested previously for weight masks, it's best to start small with this value and increase it until the accuracy of the model drops. 

We also need to copy over the weights from the previous model.

```
# Build and compile model
module_model.build((None,784))
module_model.compile(loss="categorical_crossentropy", optimizer="adam")

module_model.get_layer('dense_300').set_weights(model.get_layer('dense_300').get_weights())
module_model.get_layer('dense_100').set_weights(model.get_layer('dense_100').get_weights())
```

Because we want to isolate the module in the network responsible for identifying zeros, we re-format the labels to reflect the binary classifciation task of predicting either the existence of a zero in the image or its absence.

```
# Prepare the binary data
binary_y_train = np.column_stack((np.sum(y_train[:,[0]],axis=1),
                                  np.sum(y_train[:,list(set(range(10))-set([0]))],axis=1)))
binary_y_test = np.column_stack((np.sum(y_test[:,[0]],axis=1),
                                 np.sum(y_test[:,list(set(range(10))-set([0]))],axis=1)))

# And class weights
class_0_weight = (1 / np.sum(binary_y_train[:,0])) * (len(binary_y_train) / 2.0)
class_1_weight = (1 / np.sum(binary_y_train[:,1])) * (len(binary_y_train) / 2.0)
```

The class weights should represent the proportion of examples in each class. Finally, we can build, compile, and fit the model.

```
# Fit the module model
module_model.fit(x_train,binary_y_train,validation_data=(x_test,binary_y_test),epochs=10,
              class_weight={0:class_0_weight,1:class_1_weight})
```

The accuracy of the module is 98% and only contains 122 units.