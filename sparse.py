from scipy.sparse import coo_matrix, csr_matrix
import tensorflow as tf
import numpy as np


tf.enable_eager_execution()

X = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0.5], [0, 1, 1]])
print(X)
_, nb_features = X.shape

X_sp = coo_matrix(X)
print(X_sp.row, X_sp.col, X_sp.data)

X_tf_ind = tf.SparseTensor(indices=np.column_stack((X_sp.row, X_sp.col)), values=X_sp.col, dense_shape=X.shape)
X_tf_val = tf.SparseTensor(indices=np.column_stack((X_sp.row, X_sp.col)), values=X_sp.data, dense_shape=X.shape)

embedding_size = 20
V = tf.constant(np.random.random((nb_features, embedding_size)))

print('Truth')
print(X @ V)

result = tf.nn.embedding_lookup_sparse(V, X_tf_ind, X_tf_val, combiner='sum')
print('Test')
print(result)
