import tensorflow as tf
import data_utils
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import random
import numpy as np

class EncDecModel():

    def __init__(self, args):
        self.args = args
        # Parameters for the model
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.time_steps = args.time_steps
        self.hidden_features = args.hidden_features
        self.output_classes = args.output_classes
        self.num_layers = args.num_layers

        # The below are made in build_model and are needed for training.
        self.X = None
        self.Y = None
        self.hypothesis_index = None
        self.optimizer = None
        self.accuracy = None
        self.acc_summary = None

        # build the model
        self.build_model()

    def build_model(self):
        # Placeholders for our input data, hidden layer, and y values
        self.X = tf.placeholder("float", [None, self.time_steps, self.input_size])
        hidden_state = tf.placeholder("float", [None, self.hidden_features], name="Hidden")
        self.Y = tf.placeholder("float", [None, self.output_classes], name="Output")

        # Weights adn Biases for hidden layer and output layer
        W_hidden = tf.Variable(tf.random_normal([self.input_size,self.hidden_features]))
        W_out = tf.Variable(tf.random_normal([self.hidden_features,self.output_classes]))
        b_hidden = tf.Variable(tf.random_normal([self.hidden_features]))
        b_out = tf.Variable(tf.random_normal([self.output_classes]))

        # The Formula for the Model
        input_ = tf.reshape(self.X, [-1, self.input_size])
        lstm_cell = tf.nn.rnn_cell.GRUCell(self.hidden_features)
        input_2 = tf.split(0, self.time_steps, input_)
        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* self.num_layers, state_is_tuple=True)
        hidden_state = cells.zero_state(self.batch_size, tf.float32)
        outputs, state = seq2seq.basic_rnn_seq2seq(input_2, hidden_state, cells) # this is the black magic
        hypothesis = tf.matmul(outputs[-1], W_out) + b_out
        self.hypothesis_index = tf.argmax(hypothesis,1)

        # Define our cost and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        # Define our model evaluator
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.acc_summary = tf.scalar_summary("Accuracy", self.accuracy)

