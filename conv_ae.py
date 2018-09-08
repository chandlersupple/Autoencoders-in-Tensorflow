import numpy as np
import tensorflow as tf
from keras.preprocessing import image
sess = tf.InteractiveSession()

class Autoencoder():
    
    def __init__(self):
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.x = tf.placeholder(tf.float32, [None, 4096])
        x_4D = tf.reshape(self.x, [-1, 64, 64, 1])
        
        conv1 = tf.layers.conv2d(inputs= x_4D, filters= 256, kernel_size= (3,3), padding= 'SAME')
        mp1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding= 'SAME')
        drop1 = tf.nn.dropout(mp1, self.keep_prob)
        conv2 = tf.layers.conv2d(inputs= drop1, filters= 128, kernel_size= (3,3), padding= 'SAME')
        mp2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding= 'SAME')
        drop2 = tf.nn.dropout(mp2, self.keep_prob)
        conv3 = tf.layers.conv2d(inputs= drop2, filters= 64, kernel_size= (3,3), padding= 'SAME')
        mp3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding= 'SAME')
        drop3 = tf.nn.dropout(mp3, self.keep_prob)
        conv4 = tf.layers.conv2d(inputs= drop3, filters= 16, kernel_size= (3,3), padding= 'SAME')
        self.E = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding= 'SAME')
    
        us1 = tf.image.resize_images(self.E, size= (8, 8), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.layers.conv2d(inputs= us1, filters= 64, kernel_size= (3, 3), padding= 'SAME')
        drop4 = tf.nn.dropout(conv5, self.keep_prob)
        us2 = tf.image.resize_images(drop4, size= (16, 16), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)        
        conv6 = tf.layers.conv2d(inputs= us2, filters= 128, kernel_size= (3, 3), padding= 'SAME')
        drop5 = tf.nn.dropout(conv6, self.keep_prob)
        us3 = tf.image.resize_images(drop5, size= (32, 32), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)        
        conv7 = tf.layers.conv2d(inputs= us3, filters= 256, kernel_size= (3, 3), padding= 'SAME')
        drop6 = tf.nn.dropout(conv7, self.keep_prob)
        us4 = tf.image.resize_images(drop6, size= (64, 64), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.D = tf.layers.conv2d(inputs= us4, filters= 1, kernel_size= (3, 3), padding= 'SAME')
        
        self.loss = tf.losses.mean_squared_error(x_4D, self.D)
        self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.loss)
        
        sess.run(tf.global_variables_initializer())
        
    def encoder(self, x):
        return sess.run(self.E, feed_dict= {self.x: x, ae.keep_prob: 1})
        
    def decode(self, E):
        D = sess.run(self.D, feed_dict= {self.E: E, ae.keep_prob: 1})
        D_flat = np.reshape(D, [-1, 4096])
        return [int(term) for term in D_flat[0]]
        
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
batch_xy = ae.batch(batch_size, 32)
E = ae.encoder(batch_xy)
D = ae.decode(E)

D_img = np.reshape(D, [64, 64, 1])
sample = image.array_to_img(D_img)
sample_x = image.array_to_img(np.reshape(batch_xy, [64, 64, 1]))
