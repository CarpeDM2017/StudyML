# 문지환의 폴더

import tensorflow as tf

hello = tf.constant("Hello ML!")

sess = tf.Session()

print(sess.run(hello))
