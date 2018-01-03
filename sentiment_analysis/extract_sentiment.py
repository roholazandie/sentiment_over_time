import datetime
import functools

import numpy as np
import tensorflow as tf

from preprocessing.dataset_reader import SentimentDatasetReader
from sklearn.metrics import recall_score, precision_score

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator



class ExtractSentiment(object):

    def __init__(self, batch_size, n_classes, max_seq_length, n_dimensions, n_lstm_units, n_iterations):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.max_seq_length = max_seq_length
        self.n_dimension = n_dimensions
        self.n_lstm_units = n_lstm_units
        self.n_iterations = n_iterations
        self.n_layers = 4
        self.sentiment_dataset_reader = SentimentDatasetReader(batch_size=self.batch_size, max_seq_length=max_seq_length)
        self.word_vectors = self.sentiment_dataset_reader.word_vectors
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.n_classes])
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_length])
        self.data = tf.nn.embedding_lookup(self.word_vectors, self.input_data)
        self.prediction
        self.optimize
        self.loss
        self.accuracy
        self.lstm_network


    @define_scope
    def lstm_network(self):
        #lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_lstm_units)
        #lstm_cell = tf.contrib.rnn.GRUCell(self.n_lstm_units)
        #lstm_cell = tf.contrib.rnn.LSTMCell(self.n_lstm_units)
        cells = [tf.contrib.rnn.LSTMCell(self.n_lstm_units) for _ in range(self.n_layers)]
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstm_cell, self.data, dtype=tf.float32)
        return value



    @define_scope
    def prediction(self):
        weight = tf.get_variable("weight", initializer=tf.truncated_normal([self.n_lstm_units, self.n_classes]))
        bias = tf.get_variable("bias", initializer=tf.constant(0.1, shape=[self.n_classes]))
        value = tf.transpose(self.lstm_network, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        return (tf.matmul(last, weight) + bias)


    @define_scope
    def loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))


    @define_scope
    def correct_pred(self):
        return tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))


    @define_scope
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    # @define_scope
    # def recall(self):
    #     return tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.prediction, 1))


    @define_scope
    def optimize(self):
        return tf.train.AdamOptimizer().minimize(self.loss)



    def train(self):
        train_loss_summary = tf.summary.scalar('train_loss', self.loss)
        train_accuracy_summary = tf.summary.scalar('train_accuracy', self.accuracy)
        test_loss_summary = tf.summary.scalar('test_loss', self.loss)
        test_accuracy_summary = tf.summary.scalar('test_accuracy', self.accuracy)
        #merged = tf.summary.merge_all()
        logdir = "../tensorboard1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(logdir, sess.graph)

            for i in range(self.n_iterations):
                print(i)

                next_batch_training, next_batch_train_labels = self.sentiment_dataset_reader.get_train_batch()
                prediction_value, _ = sess.run([self.prediction, self.optimize], {self.input_data: next_batch_training,
                                             self.labels: next_batch_train_labels
                                             })





                #######metric values###################

                #print(correct_pred_value)
                # prediction_value = np.argmax(prediction_value, 1)
                # labels_value = np.argmax(next_batch_train_labels, 1)
                # precision_value = precision_score(y_true=labels_value, y_pred=prediction_value)
                # recall_value = recall_score(y_true=labels_value, y_pred=prediction_value)
                # #print(precision_value)
                # print(recall_value)



                ########test#################
                next_batch_test, next_batch_test_labels = self.sentiment_dataset_reader.get_test_batch()
                # Write summary to Tensorboard
                if (i % 50 == 0):
                    train_loss_summary_value = sess.run(train_loss_summary, {self.input_data: next_batch_training, self.labels: next_batch_train_labels})
                    train_accuracy_summary_value = sess.run(train_accuracy_summary, {self.input_data: next_batch_training, self.labels: next_batch_train_labels})
                    test_accuracy_summary_value = sess.run(test_accuracy_summary, {self.input_data: next_batch_test, self.labels: next_batch_test_labels})
                    test_loss_summary_value = sess.run(test_loss_summary, {self.input_data: next_batch_test, self.labels: next_batch_test_labels})
                    writer.add_summary(train_loss_summary_value, i)
                    writer.add_summary(train_accuracy_summary_value, i)
                    writer.add_summary(test_accuracy_summary_value, i)
                    writer.add_summary(test_loss_summary_value, i)

                if (i % 1000 == 0 and i != 0):
                    save_path = saver.save(sess, "../models1/pretrained_lstm.ckpt", global_step=i)
                    print("saved to %s" % save_path)

            writer.close()


    def test(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('../models'))
            accuracy_values = []
            for i in range(self.n_iterations):
                next_batch_test, next_batch_test_labels = self.sentiment_dataset_reader.get_test_batch()
                #next_batch_test, next_batch_test_labels = self.sentiment_dataset_reader.get_train_batch()
                accuracy_value = sess.run(self.accuracy, {self.input_data: next_batch_test, self.labels: next_batch_test_labels})
                print(accuracy_value)
                accuracy_values.append(accuracy_value)


        print("----------")
        print(np.mean(accuracy_values))
        print(np.max(accuracy_values))
        print(np.min(accuracy_values))



    def classify_sentence_sentiments(self, sentence):
        tf.reset_default_graph()
        input_sentence = self.sentiment_dataset_reader.convert_sentence_to_word_index_vector(sentence)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('../models'))
            prediction_value = sess.run(self.prediction, {self.input_data: input_sentence})

        sentence_prediction = np.argmax(prediction_value[0])
        return 'positive' if sentence_prediction == 0 else 'negative'


    def classify_batch_sentences_sentiments(self, sentences):
        #tf.reset_default_graph()
        input_sentence = self.sentiment_dataset_reader.convert_batch_sentences_to_word_index_vector(sentences)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('../models'))
            prediction_values = sess.run(self.prediction, {self.input_data: input_sentence})

        sentence_predictions = np.argmax(prediction_values, axis=1)
        sentence_predictions_labels = ['positive' if sentence_prediction == 0 else 'negative' for sentence_prediction in sentence_predictions]
        return sentence_predictions_labels




if __name__ == "__main__":
    batch_size = 100#24#1000
    max_seq_length = 30#250#50
    n_iterations = 40000
    n_lstm_units = 70#3#6#64#12
    n_dimensions = 300
    extract_sentiment = ExtractSentiment(batch_size=batch_size,
                                         n_classes=2,
                                         max_seq_length=max_seq_length,
                                         n_dimensions=n_dimensions,
                                         n_lstm_units=n_lstm_units,
                                         n_iterations=n_iterations)

    extract_sentiment.train()
    #extract_sentiment.test()
    #sentiment = extract_sentiment.classify_sentence_sentiments("worst movie ever")
    #print(sentiment)