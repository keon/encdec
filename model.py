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
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.time_steps = args.time_steps
        self.hidden_features = args.hidden_features
        self.output_classes = args.output_classes
        self.num_layers = args.num_layers

        # The below are made in build_model and are needed for training.
        self.input_var = None
        self.y_var = None
        self.hypothesis_index = None
        self.optimizer = None
        self.accuracy = None
        self.acc_summary = None

        # build the model
        self.build_model()

    def build_model(self):
        # Placeholders for our input data, hidden layer, and y values
        self.input_var = tf.placeholder("float", [None, self.time_steps, self.input_size])
        hidden_state = tf.placeholder("float", [None, self.hidden_features], name="Hidden")
        self.y_var = tf.placeholder("float", [None, self.output_classes], name="Output")

        # Weights adn Biases for hidden layer and output layer
        W_hidden = tf.Variable(tf.random_normal([self.input_size,self.hidden_features]))
        W_out = tf.Variable(tf.random_normal([self.hidden_features,self.output_classes]))
        b_hidden = tf.Variable(tf.random_normal([self.hidden_features]))
        b_out = tf.Variable(tf.random_normal([self.output_classes]))

        # The Formula for the Model
        input_ = tf.reshape(self.input_var, [-1, self.input_size])
        lstm_cell = tf.nn.rnn_cell.GRUCell(self.hidden_features)
        input_2 = tf.split(0, self.time_steps, input_)
        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* self.num_layers, state_is_tuple=True)
        hidden_state = cells.zero_state(self.batch_size, tf.float32)
        outputs, state = seq2seq.basic_rnn_seq2seq(input_2, hidden_state, cells) # this is the black magic
        hypothesis = tf.matmul(outputs[-1], W_out) + b_out
        self.hypothesis_index = tf.argmax(hypothesis,1)

        # Define our cost and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis,self.y_var))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(cost)

        # Define our model evaluator
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(self.y_var,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.acc_summary = tf.scalar_summary("Accuracy", self.accuracy)

    def get_batch(self, data_set):
        """
        Args:
          data: lists of pairs of input and output data that we use to create a batch.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.input_size, self.output_classes
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
          encoder_input, decoder_input = random.choice(data_set)

          # Encoder inputs are padded and then reversed.
          encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
          encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

          # Decoder inputs get an extra "GO" symbol, and are padded then.
          decoder_pad_size = decoder_size - len(decoder_input) - 1
          decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        print(encoder_inputs[0], decoder_inputs[0])

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
          batch_encoder_inputs.append(
              np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
          batch_decoder_inputs.append(
              np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

          # Create target_weights to be 0 for targets that are padding.
          batch_weight = np.ones(self.batch_size, dtype=np.float32)
          for batch_idx in xrange(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
              target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
              batch_weight[batch_idx] = 0.0
          batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self, sess, encoder_inputs, decoder_inputs, target_weights):
        feed = {self.input_var: encoder_inputs, self.y_var: decoder_inputs}
        sess.run(self.optimizer, feed_dict=feed)
        if i % args.display_step == 0:
            # Calculate Accuracy
            summary, acc = sess.run([self.acc_summary, self.accuracy], feed_dict=feed)
            print "Step: " + str(i) + ", Training Accuracy: " + str(acc)
            train_writer.add_summary(summary, i)
        if i % 100 == 0 and not(i==0):
            seq = ''
            x_inp = batch_x
            print("batchx shape: ", batch_x.shape)
            print("batchy shape: ", batch_y.shape)
            for j in range(140):
                index = self.hypothesis_index.eval({
                    self.input_var: x_inp,
                    self.y_var: batch_y
                })
                next_letter = unichr(index[0])
                x_inp = source[i+0+1+j:i+args.batch_size*args.time_steps*args.input_size+1+j]
                x_inp[-1] = float(ord(next_letter))
                x_inp = x_inp.reshape((args.batch_size,args.time_steps,args.input_size))
                seq += next_letter
            f = open('save/gen' + str(i) + '.txt', 'w+')
            print "save:\n" +seq
            f.write(seq)
            f.close()

