from scipy.sparse import csr_matrix
import tensorflow as tf
import numpy as np
# import time


# print('Start')
# dt = time.time()

# Handle data
# X_data = tf.data.Dataset.from_tensor_slices(X_fm_train)
# y_data = tf.data.Dataset.from_tensor_slices(y_train)
# user_data = tf.data.Dataset.from_tensor_slices(X_train[:, 0])
# item_data = tf.data.Dataset.from_tensor_slices(X_train[:, 1])
# dataset = tf.data.Dataset.zip((X_data, y_data, user_data, item_data)).batch(batch_size)
# iterator = dataset.make_initializable_iterator()
# X_fm_batch, outcomes, user_batch, item_batch = iterator.get_next()

# print('Stop', time.time() - dt)  # 16 seconds pour Movielens

# sess.run(iterator.initializer)

# x_batch, y_batch = sess.run(next_element)

M = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 2], [0, 1, 1]])

# print(M)
# S = csr_matrix(M).tocoo()
# print('deb', S.row, S.col)

# First observation: these two slicing operations provide different orderings
S = csr_matrix(M)[1:3].tocoo()
print('1:3', S.row, S.col)
S = csr_matrix(M)[[1, 2]].tocoo()
print('1,2', S.row, S.col)

# X_fm = load_npz('data/movie100k/X_fm.npz')
# S = X_fm[[1, 2], :].tocoo()
# print(S.row, S.col, S.data, S.shape)

entries = np.column_stack((S.row, S.col, S.data))
nb_entries = len(S.data)
ordering = np.arange(nb_entries)

print('1', ordering)
# print(entries[ordering])
# ordering = np.lexsort((S.col, S.row))
print('2', ordering)
# print(entries[ordering])

# print(entries[ordering, 0], entries[ordering, 1], entries[ordering, 2])
# S = X_fm[[1, 2], :].tocoo()
# rows, cols, data = find(S)
# print(rows, cols, data)
# S = X_fm[1:3, :].tocoo()
# print(S.row, S.col, S.data, S.shape)
X_train = tf.SparseTensor(indices=entries[ordering, :2],
                           values=entries[ordering, 2], dense_shape=S.shape)
y_train = np.random.random(len(S.data))
# batch_size = 2
# nb_epochs = 2
#.batch(batch_size)

X_data = tf.data.Dataset.from_tensor_slices(X_train)
y_data = tf.data.Dataset.from_tensor_slices(y_train)
dataset = tf.data.Dataset.zip((X_data, y_data))
iterator = dataset.make_initializable_iterator()
X_sample, y_sample = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run([X_sample, y_sample]))
    # while True:
    #     try:
    #         break
    #     except tf.errors.OutOfRangeError:
    #         break
