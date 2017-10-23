import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import ae_tools as tools


class AutoEncoder:

    def __init__(self, n_input, n_hidden, optimizer=tf.train.AdamOptimizer(learning_rate=0.001)):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation = tf.nn.softplus
        self.weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.activation(tf.add(tf.matmul(self.x, self.weights["w1"]), self.weights["b1"]))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights["w2"]), self.weights["b2"])

        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        # sess
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights["w1"] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        all_weights["b1"] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights["w2"] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights["b2"] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run([self.cost, self.optimizer], feed_dict={self.x: X})
        return cost

    def calculate_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden=None, batch_size=10):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([batch_size, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def get_weights(self):
        return self.sess.run(self.weights["w1"])

    def get_biases(self):
        return self.sess.run(self.weights["b1"])

    pass


class Runner:

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.mnist = input_data.read_data_sets("data/", one_hot=True)
        self.x_train, self.x_test = tools.min_max_scale(self.mnist.train.images, self.mnist.test.images)
        self.train_number = self.mnist.train.num_examples

    def train(self, train_epochs=20, batch_size=64, display_step=1):
        for epoch in range(train_epochs):
            avg_cost = 0.
            total_batch = int(self.train_number) // batch_size
            for i in range(total_batch):
                batch_xs = tools.get_random_block_from_data(self.x_train, batch_size)
                cost = self.autoencoder.partial_fit(batch_xs)
                avg_cost += cost / self.train_number * batch_size

            if epoch % display_step == 0:
                self.save_result(file_name="{}-result-{}-{}-{}-{}".format(1, epoch, self.autoencoder.n_input, self.autoencoder.n_hidden, avg_cost))
                print(time.strftime("%H:%M:%S", time.localtime()), "Epoch:{}".format(epoch + 1), "cost={:.9f}".format(avg_cost))

        print(time.strftime("%H:%M:%S", time.localtime()), "Total cost: {}".format(self.autoencoder.calculate_total_cost(self.mnist.test.images)))

    def save_result(self, file_name, n_show=10):
        # 显示编码结果和解码后结果
        images = tools.get_random_block_from_data(self.x_test, n_show, fixed=True)
        encode = self.autoencoder.transform(images)
        decode = self.autoencoder.generate(encode)

        # 对比原始图片重建图片
        tools.save_result(images, encode, decode, save_path="result/ae2/{}.jpg".format(file_name))


if __name__ == '__main__':
    runner = Runner(autoencoder=AutoEncoder(n_input=784, n_hidden=80))
    runner.train()

    pass
