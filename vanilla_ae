import numpy as np
import tensorflow as tf
from keras.preprocessing import image
sess = tf.InteractiveSession()

class Autoencoder():
    
    def __init__(self):
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.x = tf.placeholder(tf.float32, [None, 4096])
        
        L1 = tf.layers.dense(self.x, 1024)
        drop1 = tf.nn.dropout(L1, self.keep_prob)
        L2 = tf.layers.dense(drop1, 512)
        drop2 = tf.nn.dropout(L2, self.keep_prob)
        L3 = tf.layers.dense(drop2, 256)
        drop3 = tf.nn.dropout(L3, self.keep_prob)
        L4 = tf.layers.dense(drop3, 64)
        drop4 = tf.nn.dropout(L4, self.keep_prob)
        self.E = tf.layers.dense(drop4, 32)
    
        L5 = tf.layers.dense(self.E, 64)
        drop5 = tf.nn.dropout(L5, self.keep_prob)
        L6 = tf.layers.dense(drop5, 256)
        drop6 = tf.nn.dropout(L6, self.keep_prob)
        L7 = tf.layers.dense(drop6, 512)
        drop7 = tf.nn.dropout(L7, self.keep_prob)
        L8 = tf.layers.dense(drop7, 1024)
        drop8 = tf.nn.dropout(L8, self.keep_prob)
        self.D = tf.layers.dense(drop8, 4096)
        
        self.loss = tf.losses.mean_squared_error(self.x, self.D)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
        sess.run(tf.global_variables_initializer())
        
    def encoder(self, x):
        return sess.run(self.E, feed_dict= {self.x: x, ae.keep_prob: 1})
        
    def decode(self, E):
        D = sess.run(self.D, feed_dict= {self.E: E, ae.keep_prob: 1})
        return [int(term) for term in D[0]]
        
    def batch(self, batch_size, batch_iter):
        batch_iarr = []
        for inst in range (batch_iter * batch_size + 1, batch_iter * batch_size + batch_size + 1):
            image_inst = image.load_img(('dog.%s.jpg' %(inst)), color_mode= 'grayscale', target_size = [64, 64])
            inst_arr = image.img_to_array(image_inst)
            inst_flat = np.reshape(inst_arr, [4096])
            batch_iarr.append(inst_flat)
        return batch_iarr

batch_size = 32
batches_in_epoch = 4000 // batch_size
ae = Autoencoder()

for epoch in range (128):
    for batch_iter in range (batches_in_epoch):
        batch_xy = ae.batch(batch_size, batch_iter)
        sess.run(ae.optimizer, feed_dict= {ae.x: batch_xy, ae.keep_prob: 0.5})
        if batch_iter % 25 == 0:
            error = sess.run(ae.loss, feed_dict= {ae.x: batch_xy, ae.keep_prob: 1})
            print('Epoch: %s, Batch %s/%s, Loss: %s' %(epoch, batch_iter, batches_in_epoch, error))

batch_size = 1
batch_xy = ae.batch(batch_size, 0)
E = ae.encoder(batch_xy)
D = ae.decode(E)
D_img = np.reshape(D, [64, 64, 1])
sample = image.array_to_img(D_img)
sample_x = image.array_to_img(np.reshape(batch_xy, [64, 64, 1]))
