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
    p.add_argument('--lr', type=float, default=1e-3)
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

def train(args):
    print("Getting Dataset...")
    source, target = data_loader.get_data(args.data_dir) # TODO make target data
    print("Initializing the Model...")
    model = EncDecModel(args)

    print("Starting training...")
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        proc = subprocess.Popen(["tensorboard","--logdir="+address])

        for i in range(args.epoches):
            batch_x = source[i+0 : i+args.batch_size * args.time_steps * args.input_size]
            charlist = source[i+1 : i+args.batch_size+1]
            batch_y = np.zeros((len(charlist), args.output_classes))
            for j in range(len(charlist)-1):
                batch_y[j][int(charlist[j])] = 1.0
            print("batch_x shape:", batch_x.shape, "batch_y shape", batch_y.shape)
            # Reshape batch to input size
            batch_x = batch_x.reshape((args.batch_size, args.time_steps, args.input_size))

            print("batch_x shape:", batch_x.shape, "batch_y shape", batch_y.shape)
            feed = {model.X: batch_x, model.Y: batch_y}
            sess.run(model.optimizer, feed_dict=feed)
            if i % args.display_step == 0:
                # Calculate Accuracy
                summary, acc = sess.run([model.acc_summary, model.accuracy], feed_dict=feed)
                print "Step: " + str(i) + ", Training Accuracy: " + str(acc)
            if i % 100 == 0 and not(i==0):
                seq = ''
                x_inp = batch_x
                print("batchx shape: ", batch_x.shape)
                print("batchy shape: ", batch_y.shape)
                for j in range(140):
                    index = model.hypothesis_index.eval({
                        model.X: x_inp,
                        model.Y: batch_y
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
            if i % 1000 == 0 and not(i == 0):
                saver.save(sess,"save/" +str(i) +".ckpt")


        print "Training is COMPLETE!"
        x_test = source[i:i+args.batch_size]
        y_test = source[i*args.batch_size+1:i*2*args.batch_size]
        test_accuracy = sess.run(model.accuracy, feed_dict=feed)
        print ("Final test accuracy: %g" %(test_accuracy))
        saver.save(sess,"save/model.ckpt")
        proc.kill()

if __name__ == "__main__":
    main()
