import tensorflow as tf
import numpy as np
from keras.preprocessing import image
sess = tf.InteractiveSession()

class Autoencoder():
    
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 4096])
        x_4D = tf.reshape(self.x, [-1, 64, 64, 1])
        
        conv1 = tf.layers.conv2d(inputs= x_4D, filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= 'SAME')
        conv2 = tf.layers.conv2d(inputs= conv1, filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= 'SAME')
        conv3 = tf.layers.conv2d(inputs= conv2, filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= 'SAME')
        flat_convE = tf.contrib.layers.flatten(conv3)
        self.E = tf.layers.dense(flat_convE, 64, activation= tf.nn.relu)
        
        resh_E = tf.reshape(self.E, [-1, 8, 8, 1])
        us1 = tf.image.resize_images(resh_E, size= (8, 8), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv4 = tf.layers.conv2d(inputs= us1, filters= 64, kernel_size= (5, 5), padding= 'SAME')
        us2 = tf.image.resize_images(conv4, size= (22, 22), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.layers.conv2d(inputs= us2, filters= 64, kernel_size= (5, 5), padding= 'SAME')
        us3 = tf.image.resize_images(conv5, size= (64, 64), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.D = tf.layers.conv2d(inputs= us3, filters= 1, kernel_size= (5, 5), padding= 'SAME')
        
        self.loss = tf.losses.mean_squared_error(x_4D, self.D)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
        sess.run(tf.global_variables_initializer())
        
    def encoder(self, x):
        return sess.run(self.E, feed_dict= {self.x: x})
        
    def decode(self, E):
        D = sess.run(self.D, feed_dict= {self.E: E})
        D_flat = np.reshape(D, [-1, 4096])
        return [int(term) for term in D_flat[0]]
        
    def batch(self, batch_size, batch_iter):
        batch_iarr = []
        for inst in range (batch_iter * batch_size + 1, batch_iter * batch_size + batch_size + 1):
            image_inst = image.load_img(('Img-%s.jpg' %(inst)), color_mode= 'grayscale', target_size = [64, 64])
            # Modify the above line of code
            inst_arr = image.img_to_array(image_inst)
            inst_flat = np.reshape(inst_arr, [4096])
            batch_iarr.append(inst_flat)
        return batch_iarr

batch_size = 32
batches_in_epoch = 13000 // batch_size
ae = Autoencoder()

for epoch in range (128):
    for batch_iter in range (batches_in_epoch):
        batch_xy = ae.batch(batch_size, batch_iter)
        sess.run(ae.optimizer, feed_dict= {ae.x: batch_xy})
        if batch_iter % 15 == 0:
            error = sess.run(ae.loss, feed_dict= {ae.x: batch_xy})
            print('Epoch: %s, Batch %s/%s, Loss: %s' %(epoch, batch_iter, batches_in_epoch, error))

batch_size = 1
batch_xy = ae.batch(batch_size, 32)
E = ae.encoder(batch_xy)
D = ae.decode(E)

D_img = np.reshape(D, [64, 64, 1])
sample = image.array_to_img(D_img)
sample_x = image.array_to_img(np.reshape(batch_xy, [64, 64, 1]))
