import tensorflow as tf


def fetch_relevant_mean(score, seq_len, normalizer=None):
    this_max_seq_len = tf.shape(score)[0]
    relevant_score = score * tf.transpose(tf.sequence_mask(seq_len, maxlen=this_max_seq_len, dtype=tf.float32))
    nb_nonzero = tf.cast(tf.reduce_sum(seq_len), tf.float32)
    sum_scores = tf.reduce_sum(relevant_score)
    if normalizer is None:
        normalizer = nb_nonzero  # Error by sequence step (play down the errors on small sentences)
    # normalizer = batch_size  # Error by batch
    return sum_scores / normalizer

def fetch_relevant_sum_per_batch(score, seq_len):
    this_max_seq_len = tf.shape(score)[0]
    relevant_score = score * tf.transpose(tf.sequence_mask(seq_len, maxlen=this_max_seq_len, dtype=tf.float32))
    sum_scores = tf.reduce_sum(relevant_score, axis=0)
    return sum_scores

def compute_metrics(truth, pred_logits, seq_len):
    pred_proba = tf.nn.softmax(pred_logits)
    # print(pred_proba)
    pred_classes = tf.cast(tf.argmax(pred_proba, 2), tf.float32)
    acc = tf.cast(tf.equal(tf.cast(truth, tf.float32), pred_classes), tf.float32)
    # print(acc)
    squared_error = tf.cast(((pred_classes - truth) / 2) ** 2, tf.float32)
    # print(squared_error)
    rmse = fetch_relevant_mean(squared_error, seq_len) ** 0.5
    macc = fetch_relevant_mean(acc, seq_len)
    return acc, rmse, macc

def compute_binary_metrics(truth, pred_logits, seq_len):
    pred_proba = tf.nn.sigmoid(pred_logits)
    # print(pred_proba)
    acc = tf.cast(tf.equal(tf.cast(truth, tf.float32), tf.round(pred_proba)), tf.float32)
    # mask = tf.transpose(tf.sequence_mask(seq_len, maxlen=this_max_seq_len, dtype=tf.float32))
    # auc, update_op = tf.metrics.auc(truth, pred_proba, weights=mask)

    # squared_error = tf.cast(((pred_classes - truth) / 2) ** 2, tf.float32)
    # print(squared_error)
    # rmse = fetch_relevant_mean(squared_error, seq_len) ** 0.5
    macc = fetch_relevant_mean(acc, seq_len)
    return macc
