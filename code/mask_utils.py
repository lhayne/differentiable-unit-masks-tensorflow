import tensorflow as tf
import math

class MeanRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * tf.math.reduce_mean(x)

    def get_config(self):
        return {'alpha': float(self.alpha)}


class SumRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * tf.math.reduce_sum(x)

    def get_config(self):
        return {'alpha': float(self.alpha)}


class SumOfSigmoidsRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * tf.math.reduce_sum(tf.math.sigmoid(x))

    def get_config(self):
        return {'alpha': float(self.alpha)}


class GumbelSigmoidLayer(tf.keras.layers.Layer):
    def __init__(self,temperature=1,eps=1e-10,num_mask_samples=1):
        super(GumbelSigmoidLayer, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.num_mask_samples = num_mask_samples

    def call(self,x,batch_size=None,training=None):
        if batch_size == None or self.num_mask_samples == 1:
            # generate independent noise for each mask sample
            noise = (tf.math.log(tf.math.log(tf.random.uniform(tf.shape(x)) + self.eps) / 
                     tf.math.log(tf.random.uniform(tf.shape(x)) + self.eps)))
            sample_train_x = tf.math.sigmoid((x - noise)/self.temperature)
        else:
            # copy mask into batch_size repeats
            train_x = tf.repeat(x,repeats=batch_size,axis=0)
            
            # generate independent noise for each mask sample
            noise_shape = tf.concat([(self.num_mask_samples,),tf.shape(x)[1:]],0)
            noise = (tf.math.log(tf.math.log(tf.random.uniform(noise_shape) + self.eps) / 
                     tf.math.log(tf.random.uniform(noise_shape) + self.eps)))
            
            # apply those different masks to each part of the batch to stabilize gradients
            noise = tf.repeat(noise,repeats=tf.math.floordiv(batch_size,self.num_mask_samples),axis=0)
            sample_train_x = tf.math.sigmoid((train_x - noise)/self.temperature)
            
        # binarize the masks and stop the gradient
        binary_train_x = tf.stop_gradient(tf.where(sample_train_x > 0.5, 1.0, 0.0) - sample_train_x) + sample_train_x
        binary_test_x = tf.where(tf.math.sigmoid(x) > 0.5, 1.0, 0.0)
    
        return tf.keras.backend.in_train_phase(binary_train_x,
                                binary_test_x,
                                training=training)


class GumbelSigmoidMask(tf.keras.layers.Layer):
    """
    Mask layer which applies binary mask to post activations in a network
    using the straight through estimator.
    """
    def __init__(self, initial_keep_prob=0.9, temperature=1, 
                 eps=1e-10, alpha=0, regularizer='SumRegularizer',
                 num_mask_samples=1, **kwargs):
        super(GumbelSigmoidMask, self).__init__(**kwargs)
        self.initial_logit_mean = -tf.math.log((1/initial_keep_prob)-1) # inverse of the sigmoid
        self.temperature = temperature
        self.eps = eps
        self.alpha = alpha
        self.num_mask_samples = num_mask_samples
        if regularizer == 'SumRegularizer':
            self.regularizer = SumRegularizer(self.alpha)
        elif regularizer == 'MeanRegularizer':
            self.regularizer = MeanRegularizer(self.alpha)
        elif regularizer == 'SumOfSigmoidsRegularizer':
            self.regularizer = SumOfSigmoidsRegularizer(self.alpha)
        else:
            raise Exception('No regularizer named',regularizer)

    def build(self, input_shape):
        # Create a trainable weight kernel for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer=tf.keras.initializers.RandomNormal(mean=self.initial_logit_mean),
                                      regularizer=self.regularizer,
                                      trainable=True)
        self.ste = GumbelSigmoidLayer(self.temperature,self.eps,self.num_mask_samples)
        super(GumbelSigmoidMask, self).build(input_shape)

    def call(self, x):
        if self.num_mask_samples == 1:
            return x * self.ste(self.kernel)
        else:
            return x * self.ste(self.kernel,batch_size=tf.shape(x)[0])

    def compute_output_shape(self, input_shape):
        return input_shape


def add_masks(model,mask_type='gumbel',layer_kwargs={},include_top=False):
    """
    Applies mask layers after each trainable layer of sequential network.
    """
    new_model = tf.keras.Sequential()
    for i,l in enumerate(model.layers):
        if i == len(model.layers)-1 and include_top == False:
            new_model.add(tf.keras.layers.Dense(2,activation='softmax',name='classifier_head'))
            break
        if 'dropout' not in (l.name).lower():
            new_model.add(l)
        if len(l.trainable_weights) != 0:
            new_model.layers[-1].trainable = False
            if mask_type == 'gumbel':
                mask_layer = GumbelSigmoidMask(name='mask_'+l.name,**layer_kwargs)
            else:
                mask_layer = Mask(name='mask_'+l.name,**layer_kwargs)
            mask_layer.build(l.output_shape)
            new_model.add(mask_layer)
        elif 'dropout' not in (l.name).lower():
            new_model.layers[-1].trainable = False

    return new_model


def get_activations(model,layer_name,input):
    """
    Uses the functional API to construct subnetwork using inputs and outputs from part of full network,
    passes in the input, and returns the outputs.

    Courtesy of StackOverflow
    https://stackoverflow.com/questions/66571756/how-can-you-get-the-activations-of-the-neurons-during-an-inference-in-tensorflow
    """
    intermediate_output = tf.keras.Model(model.input, 
                                         model.get_layer(layer_name).output)
    activations = intermediate_output(input)
    tf.keras.backend.clear_session()
    return activations


def IOU(mask_0,mask_1,threshold=0.5):
    """
    Intersection over union for two 1D mask arrays based on a threshold.
    """
    mask_0_indexs_kept = (mask_0 > threshold).nonzero()[0]
    mask_1_indexs_kept = (mask_1 > threshold).nonzero()[0]
    intersection = np.intersect1d(mask_0_indexs_kept,mask_1_indexs_kept)
    return len(intersection)/len(mask_0)
