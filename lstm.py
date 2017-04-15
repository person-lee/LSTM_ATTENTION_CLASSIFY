
import tensorflow as tf
from utils import attention

class LSTM(object):
    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, attention_dim, rnn_size, num_rnn_layers, num_classes, max_grad_norm, dropout = 1., l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True):
        # define input variable
        self.keep_prob = dropout
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.num_unroll_steps = num_unroll_steps
        self.l2_reg_lambda = l2_reg_lambda
        self.max_grad_norm = max_grad_norm
        self.is_training = is_training

        self.input_data=tf.placeholder(tf.int32,[None,self.num_unroll_steps])
        self.target = tf.placeholder(tf.int64,[None])
        self.mask_x = tf.placeholder(tf.float32,[self.num_unroll_steps,None])

        #build BILSTM network
        # forward rnn
        #fw_lstm_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        #if self.is_training and self.keep_prob < 1:
        #    fw_lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
        #        fw_lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob = self.keep_prob
        #    )

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)
        # backforward rnn
        #bw_lstm_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        #if self.is_training and self.keep_prob < 1:
        #    bw_lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
        #        bw_lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob = self.keep_prob
        #    )

        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            self.embeddings = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="embeddings")
            inputs=tf.nn.embedding_lookup(self.embeddings, self.input_data)

        # dropout
        if self.is_training and self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.keep_prob)

        inputs = [tf.squeeze(input, [1]) for input in tf.split(1, self.num_unroll_steps, inputs)]

        out_put, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)

        out_put = tf.transpose(out_put, perm=[1, 0, 2])#(batch_size, steps, rnn_size*2)

        output = attention(out_put, self.attention_dim, self.l2_reg_lambda)
        #output = tf.squeeze(out_put[:, -1, :])
            
        # dropout
        if self.is_training and self.keep_prob < 1:
            output = tf.nn.dropout(output, self.keep_prob)
        #out_put = out_put * self.mask_x[:,:,None]

        #with tf.name_scope("mean_pooling_layer"):
        #    out_put = tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", initializer=tf.truncated_normal([2 * self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(0., shape=[1]))
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            #if self.l2_reg_lambda>0:
            #    l2_loss += tf.nn.l2_loss(softmax_w)
            #    l2_loss += tf.nn.l2_loss(softmax_b)
            #    weight_decay = tf.mul(l2_loss, self.l2_reg_lambda, name='l2_loss')
            #    tf.add_to_collection('losses', weight_decay)

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target)
            #tf.add_to_collection('losses', self.loss)
            #total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.target)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

        #add summary
        loss_summary = tf.scalar_summary("loss",self.cost)
        #add summary
        accuracy_summary=tf.scalar_summary("accuracy_summary",self.accuracy)

        if not is_training:
            return

        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      self.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.merge_summary(grad_summaries)

        self.summary =tf.merge_summary([loss_summary,accuracy_summary,self.grad_summaries_merged])

        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
