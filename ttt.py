import tensorflow as tf
log_dir = "/home/rohola/tmp/tutorial_log_dir"

def func1():

    a = tf.get_variable(name='a', shape=(3, 3), dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(minval=3, maxval=4))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_value = sess.run(a)
        print(a_value)
        saver.save(sess, "models/ttt.ckpt")


def func2():
    #scope.reuse_variables()
    a2 = tf.get_variable(name='a', shape=(3, 3), dtype=tf.float32,
                         initializer=tf.random_uniform_initializer(minval=3, maxval=4))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "models/ttt.ckpt")
        a2_value = sess.run(a2)
        print(a2_value)


def tes():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({'Date_Time': pd.date_range('10/1/2001 10:00:00', periods=3, freq='10H'),
                       'B': ['positive', 'negative', 'negative']})

    print(df)
    df = df.groupby([df['Date_Time'].dt.date])['B'].value_counts()
    print(df)
    a = np.random.rand(100,10)
    n=1

def dd():
    from datetime import date, timedelta

    d = date(2017,3,4) + timedelta(days=1)
    print(d)


if __name__ == "__main__":
    dd()