import argparse
import numpy as np
import datetime
import subprocess
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from model import EncDecModel
import data_loader as data_loader
import data_utils

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default="data")
    p.add_argument('--save_dir', type=str, default="save")
    p.add_argument('--alpha', type=float, default=1e-3)
    p.add_argument('--epoches', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--display_step', type=int, default=10)
    p.add_argument('--input_size', type=int, default=1)
    p.add_argument('--time_steps', type=int, default=100)
    p.add_argument('--hidden_features', type=int, default=256)
    p.add_argument('--output_classes', type=int, default=127)
    p.add_argument('--num_layers', type=int, default=3)
    args = p.parse_args()

    train(args)
    # self_test(args)


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.
    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language.
    max_size: maximum number of lines to read, all other will be ignored.
    Returns:
    data_set: list of (source, target) pairs read from the provided data
    files source and target are lists of token-ids.
    """
    data_set = []
    with open(source_path, mode="r") as source_file, \
            open(target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()][:50] # TODO: hmm
            target_ids = [int(x) for x in target.split()][:50]
            target_ids.append(data_utils.EOS_ID)
            data_set.append([source_ids, target_ids])
            source, target = source_file.readline(), target_file.readline()
    return data_set

def train(args):
    print("Getting Dataset...")
    source, target = data_loader.get_data(args.data_dir)
    # data_set = read_data('data/source.txt', 'data/target.txt', 400)
    # source, target = data_set
    print("Initializing the Model...")
    model = EncDecModel(args)

    print("Starting training...")
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        address = "/tmp/log/cooperbug/" + str(datetime.datetime.now()).replace(' ','')
        train_writer = tf.train.SummaryWriter(address,sess.graph)
        proc = subprocess.Popen(["tensorboard","--logdir="+address])

        for i in range(args.epoches):
            batch_x = source[i+0 : i+args.batch_size * args.time_steps * args.input_size]
            # batch_y = target[i+0 : i+args.batch_size * args.time_steps * args.input_size]
            charlist = source[i+1 : i+args.batch_size+1]
            batch_y = np.zeros((len(charlist), args.output_classes))
            for j in range(len(charlist)-1):
                batch_y[j][int(charlist[j])] = 1.0
            print("batch_x shape:", batch_x.shape, "batch_y shape", batch_y.shape)
            # Reshape batch to input size
            batch_x = batch_x.reshape((args.batch_size, args.time_steps, args.input_size))
            # Run an interation of training
            print("batch_x shape:", batch_x.shape, "batch_y shape", batch_y.shape)
            feed = {model.input_var: batch_x, model.y_var: batch_y}
            sess.run(model.optimizer, feed_dict=feed)
            if i % args.display_step == 0:
                # Calculate Accuracy
                summary, acc = sess.run([model.acc_summary, model.accuracy], feed_dict=feed)
                print "Step: " + str(i) + ", Training Accuracy: " + str(acc)
                train_writer.add_summary(summary, i)
            if i % 100 == 0 and not(i==0):
                seq = ''
                x_inp = batch_x
                print("batchx shape: ", batch_x.shape)
                print("batchy shape: ", batch_y.shape)
                for j in range(140):
                    index = model.hypothesis_index.eval({
                        model.input_var: x_inp,
                        model.y_var: batch_y
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
            # if i % 1000 == 0 and not(i == 0):
                # saver.save(sess,"/home/josh/Documents/Cooperbot/Models/model_iter" +str(i) +".ckpt")


        print "Training is COMPLETE!"
        # x_test = source[i:i+args.batch_size]
        # y_test = source[i*args.batch_size+1:i*2*args.batch_size]
        # test_accuracy = sess.run(model.accuracy, feed_dict=feed)
        # print ("Final test accuracy: %g" %(test_accuracy))
        # saver.save(sess,"/home/josh/Documents/Cooperbot/model.ckpt")
        # proc.kill()


def self_test(args):
    """Test the translation model."""
    print("Preparing Dataset...")
    data_set = read_data('data/source.txt', 'data/target.txt', 400)
    print("dataset:",data_set[0])
    print("Initializing the Model...")
    model = EncDecModel(args)
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
# def embedding_rnn_seq2seq(encoder_inputs,
                          # decoder_inputs,
                          # cell,
                          # num_encoder_symbols,
                          # num_decoder_symbols,
                          # embedding_size,
                          # output_projection=None,
                          # feed_previous=False,
                          # dtype=None,
                          # scope=None):
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        print("train")
        for _ in xrange(5):  # Train the fake model for 5 steps.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            data_set)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights)

if __name__ == "__main__":
    main()
