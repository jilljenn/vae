def csv_to_fm():
    df = pd.read_csv(os.path.join(PATH, 'vae/data', DATA, 'data.csv'))
    print('Starts at', df['user'].min(), df['item'].min())
    try:
        with open(os.path.join(PATH, 'vae/data', DATA, 'config.yml')) as f:
            config = yaml.load(f)
            nb_users = config['nb_users']
            nb_items = config['nb_items']
    except IOError:
        nb_users = 1 + df['user'].max()
        nb_items = 1 + df['item'].max()
    df['item'] += nb_users
    print(df.head())

    nb_entries = len(df)

    # Build sparse features
    X_fm_file = os.path.join(PATH, 'vae/data', DATA, 'X_fm.npz')
    if not os.path.isfile(X_fm_file):
        rows = np.arange(nb_entries).repeat(2)
        cols = np.array(df[['user', 'item']]).flatten()
        data = np.ones(2 * nb_entries)
        X_fm = coo_matrix((data, (rows, cols)), shape=(nb_entries, nb_users + nb_items)).tocsr()

        q_file = X_fm_file.replace('X_fm', 'q')
        if os.path.isfile(q_file):
            q = load_npz(q_file)
            X_fm = hstack((X_fm, q[df['item'] - nb_users])).tocsr()

        if is_regression:
            y_fm = np.array(df['rating'])
        else:
            y_fm = np.array(df['outcome']).astype(np.float32)
        save_npz(X_fm_file, X_fm)
        np.save(os.path.join(PATH, 'vae/data', DATA, 'y_fm.npy'), y_fm)

i = {}
i['trainval'], i['test'] = train_test_split(list(range(nb_entries)), test_size=0.2)
i['train'], i['valid'] = train_test_split(i['trainval'], test_size=0.2)
data = {key: df.iloc[i[key]] for key in {'train', 'valid', 'trainval', 'test'}}

X = {}
X_sp_ind = {}
X_sp_val = {}
tf_dataset = {}
init_op = {}
y = {}
nb_samples = {}
nb_occurrences = {
    'train': X_fm[i['train']].sum(axis=0).A1,
    'trainval': X_fm[i['trainval']].sum(axis=0).A1
}
nb_iters = options.nb_batches

for category in data:
    X[category] = np.array(data[category][['user', 'item']])
    print(category, X[category].size)
    if is_regression:
        y[category] = np.array(data[category]['rating']).astype(np.float32)
    else:
        y[category] = np.array(data[category]['outcome']).astype(np.float32)
    nb_samples[category] = len(X[category])
    S = X_fm[i[category]].tocoo()
    # Have to sort the indices properly, otherwise tf.data will cry (reported bug)
    entries = np.column_stack((S.row, S.col, S.data))
    indices = np.lexsort((S.col, S.row))
    # Build TF-specific datasets
    X_sp_ind[category] = tf.SparseTensor(indices=entries[indices, :2], values=entries[indices, 1], dense_shape=S.shape)
    X_sp_val[category] = tf.SparseTensor(indices=entries[indices, :2], values=entries[indices, 2], dense_shape=S.shape)
    X_ind_data = tf.data.Dataset.from_tensor_slices(X_sp_ind[category])
    X_val_data = tf.data.Dataset.from_tensor_slices(X_sp_val[category])
    y_data = tf.data.Dataset.from_tensor_slices(tf.constant(y[category]))
    if 'train' in category:
        batch_size = nb_samples[category] // nb_iters
        tf_dataset[category] = tf.data.Dataset.zip((X_ind_data, X_val_data, y_data)).shuffle(10000).batch(batch_size)
    else:
        tf_dataset[category] = tf.data.Dataset.zip((X_ind_data, X_val_data, y_data)).batch(nb_samples[category])
