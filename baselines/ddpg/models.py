import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name, share_first_layer):
        self.name = name
        self.share_first_layer = share_first_layer

    @property
    def vars(self):
        if self.name=='actor' or self.name=='critic':
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='share')+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        if self.name == 'actor' or self.name == 'critic':
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='share')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        else:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, share_first_layer=False):
        super(Actor, self).__init__(name=name, share_first_layer=share_first_layer)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        if self.name=='actor' and self.share_first_layer:
            with tf.variable_scope("share", reuse=tf.AUTO_REUSE):
                x = obs
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            if not (self.name=='actor' and self.share_first_layer):
                x = obs
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, share_first_layer=False):
        super(Critic, self).__init__(name=name, share_first_layer=share_first_layer)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        if self.name=='critic' and self.share_first_layer:
            with tf.variable_scope("share", reuse=tf.AUTO_REUSE):
                x = obs
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=-1)


        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            if not (self.name=='critic' and self.share_first_layer):
                x = obs
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
