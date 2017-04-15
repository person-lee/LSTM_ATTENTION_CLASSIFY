# coding=utf-8
import time
import logging
import tensorflow as tf
import datetime
import os

import numpy as np

from lstm import LSTM
from data_helper import load_data, load_embedding, batch_iter, create_valid, build_vocab, load_label

#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "data/train.txt", "train corpus file")
tf.flags.DEFINE_string("test_file", "data/test.txt", "test corpus file")
tf.flags.DEFINE_string("word_file", "data/words.txt", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "data/vectors.txt", "vector file")
tf.flags.DEFINE_string("label_file", "data/cateAndQuest.txt", "label file")
tf.flags.DEFINE_integer("rnn_size", 201, "rnn size of lstm")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "the number of rnn layer")
tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_integer("attention_dim", 100, "embedding size")
tf.flags.DEFINE_integer("sequence_len", 80, "embedding size")
tf.flags.DEFINE_float("dropout", 0.5, "the proportion of dropout")
tf.flags.DEFINE_float("max_grad_norm", 5, "the max of gradient")
tf.flags.DEFINE_float("init_scale", 0.1, "initializer scale")
tf.flags.DEFINE_integer("batch_size", 128, "batch size of each batch")
tf.flags.DEFINE_float('lr',0.1,'the learning rate')
tf.flags.DEFINE_float('lr_decay',0.6,'the learning rate decay')
tf.flags.DEFINE_integer("epoches", 100, "epoches")
tf.flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
tf.flags.DEFINE_integer("evaluate_every", 1000, "run evaluation")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "run evaluation")
tf.flags.DEFINE_integer("l2_reg_lambda", 0.01, "l2 regulation")
tf.flags.DEFINE_string("out_dir", "save/", "output directory")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.9, "use memory rate")

FLAGS = tf.flags.FLAGS
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger ---------------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log", mode="w")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)
#----------------------------- define a logger end -----------------------------------

#------------------------------- evaluate model -----------------------------------
def evaluate(model, session, data, global_steps=None):
    correct_num=0
    total_num=len(data)
    for step, batch in enumerate(batch_iter(data, batch_size=FLAGS.batch_size)):
        x, y, mask_x = zip(*batch)
        fetches = model.correct_num
        feed_dict={
            model.input_data:x,
            model.target:y,
            model.mask_x:np.transpose(mask_x)
        }
        
        count=session.run(fetches, feed_dict)
        correct_num += count

    accuracy=float(correct_num)/total_num
    return accuracy
#------------------------------ evaluate model end -------------------------------------

#----------------------------- run epoch -----------------------------------

def run_epoch(model,session,data,global_steps,valid_model,valid_data,train_summary_writer):
    for step, batch in enumerate(batch_iter(data, batch_size = FLAGS.batch_size)):
        x, y, mask_x = zip(*batch)
        feed_dict={
            model.input_data:x,
            model.target:y,
            model.mask_x:np.transpose(mask_x)
        }
        
        fetches = [model.cost, model.accuracy, model.train_op, model.summary]
        cost, accuracy, _, summary = session.run(fetches, feed_dict)

        train_summary_writer.add_summary(summary,global_steps)
        train_summary_writer.flush()

        timestr = datetime.datetime.now().isoformat()
        logging.info("%s, the %i step, train cost is:%f and the train accuracy is %6.7f"%(timestr, global_steps, cost, accuracy))
        if(global_steps % FLAGS.evaluate_every == 0):
            valid_accuracy = evaluate(valid_model,session,valid_data,global_steps)
            logging.info("%s, the valid accuracy is %f"%(timestr, valid_accuracy))

        global_steps += 1

    return global_steps
#---------------------------- run epoch end -------------------------------------


#------------------------------------load data -------------------------------
word2idx, idx2word = build_vocab(FLAGS.word_file)
label2idx, idx2label = load_label(FLAGS.label_file)
train_x, train_y, train_mask = load_data(FLAGS.train_file, word2idx, label2idx, FLAGS.sequence_len)
logging.info("load train data finish")
train_data, valid_data = create_valid(zip(train_x, train_y, train_mask))
num_classes = len(label2idx)
embedding = load_embedding(FLAGS.embedding_size, filename=FLAGS.embedding_file)
test_x, test_y, test_mask = load_data(FLAGS.test_file, word2idx, label2idx, FLAGS.sequence_len)
logging.info("load test data finish")
#----------------------------------- load data end ----------------------

#----------------------------------- execute train ---------------------------------------
with tf.Graph().as_default():
    with tf.device("/cpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            initializer = tf.random_uniform_initializer(-1 * FLAGS.init_scale, 1 * FLAGS.init_scale)
            with tf.variable_scope("model", reuse = None, initializer = initializer):
                model = LSTM(FLAGS.batch_size, FLAGS.sequence_len, embedding, FLAGS.embedding_size, FLAGS.attention_dim, FLAGS.rnn_size, FLAGS.num_rnn_layers, num_classes, FLAGS.max_grad_norm, dropout = FLAGS.dropout, is_training=True)

            with tf.variable_scope("model", reuse = True, initializer = initializer):
                valid_model = LSTM(FLAGS.batch_size, FLAGS.sequence_len, embedding, FLAGS.embedding_size, FLAGS.attention_dim, FLAGS.rnn_size, FLAGS.num_rnn_layers, num_classes, FLAGS.max_grad_norm, is_training=False)
                test_model = LSTM(FLAGS.batch_size, FLAGS.sequence_len, embedding, FLAGS.embedding_size, FLAGS.attention_dim, FLAGS.rnn_size, FLAGS.num_rnn_layers, num_classes, FLAGS.max_grad_norm, is_training=False)

            #add summary
            train_summary_dir = os.path.join(FLAGS.out_dir,"summaries","train")
            train_summary_writer =  tf.train.SummaryWriter(train_summary_dir,sess.graph)

            #add checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())


            tf.initialize_all_variables().run()
            global_steps=1
            begin_time=int(time.time())

            for i in range(FLAGS.epoches):
                logging.info("the %d epoch training..."%(i+1))
                #lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_decay_epoch, 0.0)
                #model.assign_new_lr(sess, FLAGS.lr*lr_decay)
                model.assign_new_lr(sess, 1e-4)
                global_steps=run_epoch(model,sess, train_data, global_steps, valid_model, valid_data, train_summary_writer)

                if i % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess,checkpoint_prefix,global_steps)
                    logging.info("Saved model chechpoint to{}\n".format(path))

            logging.info("the train is finished")
            end_time=int(time.time())
            logging.info("training takes %d seconds already\n"%(end_time-begin_time))
            test_accuracy=evaluate(test_model, sess, zip(test_x, test_y, test_mask))
            logging.info("the test data accuracy is %f"%test_accuracy)
#----------------------------------- execute train end -----------------------------------
