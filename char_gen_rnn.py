import tensorflow as tf
from text_datasets import Reuters
import text_datasets as tds
from tensorflow.contrib.rnn import BasicLSTMCell

charset_size = 256
learning_rate = 0.0001
lstm_size = 64
timesteps = 20
batch_size=100

def model_fn(features, labels, mode, params):
    output_weights = tf.Variable(tf.random_normal([lstm_size, charset_size], dtype=tf.float32))
    output_bias = tf.Variable(tf.random_normal([charset_size], dtype=tf.float32))

    input_ = tf.unstack(tf.cast(features, tf.float32), timesteps, 1)
    lstm_layer = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, input_, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], output_weights) + output_bias
    probabilities = tf.nn.softmax(logits)
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': probabilities,
            'logits': logits
            }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))

    #_, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predicted_classes, name="accuracy_op")

    #tf.summary.scalar('accuracy', accuracy_op)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={"accuracy": accuracy_op})

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    reuters = Reuters(batch_size=batch_size, timesteps=timesteps)

    (train_x, train_y), (test_x, test_y) = reuters.get_data()

    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="./char_gen")

    classifier.train(input_fn=lambda: tds.input_fn(train_x, train_y, batch_size), steps=2000)



