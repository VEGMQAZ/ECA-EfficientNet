import tensorflow as tf

class myaf(object):
    def __init__(self, x=[-2.1, 0, 2.14]):
        self.x = x
        self.input = tf.constant(self.x)

    def relu(self):
        y = tf.nn.relu(self.input)
        return y.numpy()

    def swish(self):
        y = tf.nn.swish(self.input)
        return y.numpy()

    def hswish(self):
        y = self.input * tf.nn.relu6(self.input + 3.0) / 6.0
        return y.numpy()

    def test(self):
        print(self.relu())
        print(self.swish())
        print(self.hswish())

if __name__ == '__main__':
    x = [-1.0, 2.1, -3.6, 4.9]
    a = myaf(x)
    a.test()

# 2021-04-10 guangjinzheng activate function
