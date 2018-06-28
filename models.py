import tensorflow as tf
from collections import namedtuple


KEY_POINT_MODEL_CONFIG = namedtuple("KEY_POINT_MODEL_CONFIG", ["X_PH_SHAPE", "Y_PH_SHAPE", "LEARNING_RATE"])


class KeyPointModel(object):
    def __init__(self, config):
        # hyper parameters
        self.learning_rate = tf.constant(config.LEARNING_RATE, dtype=tf.float32, name="learning_rate")

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=config.X_PH_SHAPE, name="x")
        self.y_true = tf.placeholder(tf.float32, shape=config.Y_PH_SHAPE, name="y_true")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

        # model
        self.model = self.__build_model(name="key_point_detector")

        # model functions
        self.loss_func = self.__build_loss_func()
        self.train_func = self.__build_train_func()
        self.predict_func = self.__build_predict_func()

    def __build_model(self, name="model"):
        model = {}
        with tf.name_scope(name=name):
            model["conv_1"] = tf.layers.conv2d(inputs=self.x, filters=64, kernel_size=(3, 3), padding='same',
                                               activation=tf.nn.relu, name="conv_1")

            model["max_pooling_1"] = tf.layers.max_pooling2d(inputs=model["conv_1"], pool_size=(2, 2),
                                                             strides=(2, 2), padding='same',
                                                             name="max_pooling_1")

            model["conv_1_flatten"] = tf.reshape(tensor=model["max_pooling_1"], shape=[-1, 48 * 48 * 64],
                                                 name="conv_1_flatten")

            model["dense_1"] = tf.layers.dense(inputs=model["conv_1_flatten"], units=128,
                                                activation=tf.nn.tanh, name="dense_1")

            model["dense_1_dropout"] = tf.nn.dropout(x=model["dense_1"], keep_prob=self.dropout_keep_prob,
                                                        name="dense_1_dropout")

            model["dense_2_logits"] = tf.layers.dense(inputs=model["dense_1_dropout"], units=30,
                                                        activation=None, name="dense_2_logits")

            model["dense_2_activation"] = tf.tanh(x=model["dense_2_logits"], name="dense_2_activation")

        return model

    def __build_loss_func(self):
        return tf.losses.mean_squared_error(self.model["dense_2_logits"], self.y_true)

    def __build_train_func(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.loss_func)

    def __build_predict_func(self):
        return self.model["dense_2_activation"]

    def train(self, sess, x, y, dropout_keep_prob):
        return sess.run(self.train_func, feed_dict={self.x: x,
                                                    self.y_true: y,
                                                    self.dropout_keep_prob: dropout_keep_prob})

    def predict(self, sess, x, dropout_keep_prob=1.0):
        return sess.run(self.predict_func, feed_dict={self.x: x,
                                                      self.dropout_keep_prob: dropout_keep_prob})

    def evaluate_loss(self, sess, x, y, dropout_keep_prob=1.0):
        return sess.run(self.loss_func, feed_dict={self.x: x,
                                                   self.y_true: y,
                                                   self.dropout_keep_prob: dropout_keep_prob})


class KeyPointModelWithMonitor(KeyPointModel):
    def __init__(self, config):
        super().__init__(config)
        self.__add_summary_input()
        self.__add_summary_loss()
        self.summary = self.__build_summary_func()
          
    def __add_summary_input(self):
        tf.summary.image(name="input", tensor=self.x)
        
    def __add_summary_loss(self):
        tf.summary.scalar("loss", self.loss_func)
               
    def __build_summary_func(self):
        return tf.summary.merge_all()
    
    def train(self, sess, x, y, dropout_keep_prob):
        feed_dict={self.x:x, 
                   self.y_true:y, 
                   self.dropout_keep_prob:dropout_keep_prob}
        return sess.run([self.summary, self.train_func], feed_dict=feed_dict)
    
    def evaluate_loss(self, sess, x, y, dropout_keep_prob=1.0):
        feed_dict={self.x:x, 
                   self.y_true:y, 
                   self.dropout_keep_prob:dropout_keep_prob}
        return sess.run([self.summary, self.loss_func], feed_dict=feed_dict)