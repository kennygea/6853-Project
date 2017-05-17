import tensorflow as tf, numpy as np, scipy as sp

delta = float(1e-7)
sigmoid_multiplier = float(1e7)
T = 5

h = tf.placeholder(tf.float64, name="h")

alphas = [tf.Variable(1./T, name="alpha_%s" % i, dtype=tf.float64) for i in range(T)]
weights = [tf.exp(alphas[i]) for i in range(T)]
probs = [weights[i]/sum(weights) for i in range(T)]
z = tf.pack([sum(probs[:i+1]) for i in range(T)])


inputs_minus_delta = (tf.ones((T,),dtype=tf.float64) * h - z - delta/2)/delta
inputs_plus_delta = (tf.ones((T,),dtype=tf.float64) * h - z + delta/2)/delta

f = tf.sigmoid(inputs_plus_delta * sigmoid_multiplier) * inputs_plus_delta - \
    tf.sigmoid(inputs_minus_delta * sigmoid_multiplier) * inputs_minus_delta    

f_left_shift = tf.concat(0, [tf.ones((1,),dtype=tf.float64), tf.slice(f, (0,), (T-1,))])

final_x = tf.cast(f_left_shift - f, tf.float32)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(10):
        sel = sp.random.uniform(0,1)
        print(sel)
        print(sess.run(final_x, feed_dict={
            h:sel
        }))
        


