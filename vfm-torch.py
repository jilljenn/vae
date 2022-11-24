"""
Matrix completion on toy and Movielens datasets
JJV for Deep Learning course, 2022
"""
import torch
from torch import nn, distributions
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, average_precision_score
import pandas as pd
from prepare import load_data

torch.manual_seed(42)

# LEARNING_RATE = 1
EMBEDDING_SIZE = 5
N_VARIATIONAL_SAMPLES = 1


# DATA = 'toy'
# DATA = 'movielens'
# DATA = 'movie100k'
DATA = 'movie1M'
OUTPUT_TYPE = 'reg'
if DATA.endswith('binary'):
    OUTPUT_TYPE = 'class'


if DATA == 'movie100':
    N_EPOCHS = 100
    DISPLAY_EPOCH_EVERY = 5
    BATCH_SIZE = 100
    EMBEDDING_SIZE = 3
    df_train = pd.read_csv('../Scalable-Variational-Bayesian-Factorization-Machine/data/movie100.train_libfm',
        names=('outcome', 'user', 'item'), sep=' ')
    df_train['user'] = df_train['user'].map(lambda x: x[:-2])
    df_train['item'] = df_train['item'].map(lambda x: x[:-2])
    df_train = df_train.astype(int)

    df_test = pd.read_csv('../Scalable-Variational-Bayesian-Factorization-Machine/data/movie100.test_libfm',
        names=('outcome', 'user', 'item'), sep=' ')
    df_test['user'] = df_test['user'].map(lambda x: x[:-2])
    df_test['item'] = df_test['item'].map(lambda x: x[:-2])
    df_test = df_test.astype(int)

    df = pd.concat((df_train, df_test), axis=0)
    N = df['user'].nunique()
    M = df['item'].nunique()

    X_train = torch.LongTensor(df_train[['user', 'item']].values)
    y_train = torch.LongTensor(df_train['outcome'])
    X_test = torch.LongTensor(df_test[['user', 'item']].values)
    y_test = torch.LongTensor(df_test['outcome'])

    OUTPUT_TYPE = 'class'
else:
    X_train = None
    if DATA == 'movielens':
        N_EPOCHS = 50
        DISPLAY_EPOCH_EVERY = 2
        BATCH_SIZE = 10000
        df = pd.read_csv('ml-latest-small/ratings.csv')
        films = pd.read_csv('ml-latest-small/movies.csv')
        df = df.merge(films, on='movieId')
        df['user'] = np.unique(df['userId'], return_inverse=True)[1]
        df['item'] = np.unique(df['movieId'], return_inverse=True)[1]
        N = df['user'].nunique()
        M = df['item'].nunique()
        df['item'] += N
        X = torch.LongTensor(df[['user', 'item']].to_numpy())
        y = torch.Tensor(df['rating'])
    elif DATA.startswith('movie'):
        N_EPOCHS = 50
        DISPLAY_EPOCH_EVERY = 1
        BATCH_SIZE = 100000
        # BATCH_SIZE = 100000
        '''df = pd.read_csv('data/movie100k/data.csv')#.head(1000)
                                if DATA.endswith('batch'):
                                    N_EPOCHS = 100
                                    DISPLAY_EPOCH_EVERY = 10
                                    df = df.head(1000)'''
        # films = pd.read_csv('ml-latest-small/movies.csv')
        # df = df.merge(films, on='movieId')

        N, M, X_train, X_test, y_train, y_test, _ = load_data(DATA, OUTPUT_TYPE)
        X_train = torch.LongTensor(X_train)
        nb_occ = torch.bincount(X_train.flatten())
        y_train = torch.Tensor(y_train)
        nb_train_samples = len(y_train)
        LEARNING_RATE = 1 / (1 + nb_train_samples // BATCH_SIZE)
        X_test = torch.LongTensor(X_test)
        y_test = torch.Tensor(y_test)

    elif DATA == 'movie100':
        N_EPOCHS = 100
        DISPLAY_EPOCH_EVERY = 5
        BATCH_SIZE = 1000
        EMBEDDING_SIZE = 3
        df = pd.read_parquet('data/movie100/data.parquet')
        print(df.head())
        # films = pd.read_csv('ml-latest-small/movies.csv')
        # df = df.merge(films, on='movieId')
        df['user'] = np.unique(df['userId'], return_inverse=True)[1]
        df['item'] = np.unique(df['movieId'], return_inverse=True)[1]
        N = df['user'].nunique()
        M = df['item'].nunique()
        # print(N, M, df.min(), df.max())
        # df['user'] -= 1
        df['item'] += N # - 1
        X = torch.LongTensor(df[['user', 'item']].to_numpy())
        y = torch.LongTensor(df['outcome'])        

    if X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE) # , shuffle=True


LINK = nn.functional.softplus
LINK = torch.abs


class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size, output='reg'):
        super().__init__()
        self.output = output
        self.alpha = nn.Parameter(torch.Tensor([1e9]), requires_grad=True)
        self.global_bias_mean = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        self.global_bias_scale = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_global_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_item_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_entity_prior = nn.Parameter(torch.ones(EMBEDDING_SIZE), requires_grad=True)
        self.prec_item_entity_prior = nn.Parameter(torch.ones(EMBEDDING_SIZE), requires_grad=True)
        # self.alpha = torch.Tensor([1.])
        nn.init.uniform_(self.alpha)
        
        # bias_init = torch.cat((torch.randn(N + M, 1), torch.ones(N + M, 1) * (0.02 ** 0.5)), axis=1)
        # entity_init = torch.cat((
        #     torch.randn(N + M, embedding_size),
        #     torch.ones(N + M, embedding_size) * (0.02 ** 0.5),
        # ), axis=1)
        self.bias_params = nn.Embedding(N + M, 2)#.from_pretrained(bias_init)  # w
        self.entity_params = nn.Embedding(N + M, 2 * embedding_size)#.from_pretrained(entity_init)  # V

        self.saved_global_biases = []
        self.saved_mean_biases = []
        self.saved_mean_entities = []
        self.mean_saved_global_biases = None
        self.mean_saved_mean_biases = None
        self.mean_saved_mean_entities = None

        self.global_bias_prior = distributions.normal.Normal(0, 1)
        self.bias_prior = distributions.normal.Normal(0, 1)
        self.entity_prior = distributions.normal.Normal(0, 1)
        #     torch.zeros(N + M),
        #     torch.nn.functional.softplus(torch.cat((
        #         self.prec_user_bias_prior.repeat(N),
        #         self.prec_item_bias_prior.repeat(M)
        #     )))
        # )
        # self.entity_prior = distributions.normal.Normal(
        #     torch.zeros(N + M, EMBEDDING_SIZE),
        #     torch.nn.functional.softplus(torch.cat((
        #         self.prec_user_entity_prior.repeat(N, 1),
        #         self.prec_item_entity_prior.repeat(M, 1)
        #     )))
        # )

    def save_weights(self):
        self.saved_global_biases.append(self.global_bias_mean.detach().numpy().copy())
        self.saved_mean_biases.append(self.bias_params.weight[:, 0].detach().numpy().copy())
        self.saved_mean_entities.append(self.entity_params.weight[:, :EMBEDDING_SIZE].detach().numpy().copy())
        self.mean_saved_global_biases = np.array(self.saved_global_biases).mean(axis=0)
        self.mean_saved_mean_biases = np.array(self.saved_mean_biases).mean(axis=0)
        self.mean_saved_mean_entities = np.array(self.saved_mean_entities).mean(axis=0)
        # print('size of saved', np.array(self.saved_mean_entities).shape)
        # print('test', np.array(self.saved_mean_biases)[:3, 0])

    def forward(self, x):
        uniq_entities, entity_pos, nb_occ_in_batch = torch.unique(x, return_inverse=True, return_counts=True)
        uniq_users, nb_occ_user_in_batch = torch.unique(x[:, 0], return_counts=True)
        uniq_items, nb_occ_item_in_batch = torch.unique(x[:, 1], return_counts=True)
        # nb_uniq_users = len(uniq_users)
        # nb_uniq_items = len(uniq_items)
        # print('uniq', uniq_entities.shape, 'pos', entity_pos.shape)

        # self.global_bias_prior = distributions.normal.Normal(
        #     torch.Tensor([0.]), torch.nn.functional.softplus(self.prec_global_bias_prior))
        # Global bias
        global_bias_sampler = distributions.normal.Normal(
            self.global_bias_mean,
            LINK(self.global_bias_scale)
        )
        # Biases and entities
        bias_batch = self.bias_params(x)
        entity_batch = self.entity_params(x)
        uniq_bias_batch = self.bias_params(uniq_entities)#.reshape(-1, 2)
        uniq_entity_batch = self.entity_params(uniq_entities)#.reshape(-1, 2 * EMBEDDING_SIZE)
        # print('first', bias_batch.shape, entity_batch.shape)
        # print('samplers', uniq_bias_batch.shape, uniq_entity_batch.shape)
        # scale_bias = torch.ones_like(scale_bias) * 1e-6
        bias_sampler = distributions.normal.Normal(
            uniq_bias_batch[:, 0],
            LINK(uniq_bias_batch[:, 1])
        )
        # user_bias_posterior = distributions.normal.Normal(
        #     bias_batch[:, :, 0],
        #     LINK(bias_batch[:, :, 1])
        # )
        # diag_scale_entity = nn.functional.softplus(entity_batch[:, EMBEDDING_SIZE:])
        # diag_scale_entity = torch.ones_like(diag_scale_entity) * 1e-6
        # print('scale entity', entity_batch.shape, scale_entity.shape)
        entity_sampler = distributions.normal.Normal(
            loc=uniq_entity_batch[:, :EMBEDDING_SIZE],
            scale=LINK(uniq_entity_batch[:, EMBEDDING_SIZE:])
        )
        # entity_posterior = distributions.normal.Normal(
        #     loc=entity_batch[:, :, :EMBEDDING_SIZE],
        #     scale=LINK(entity_batch[:, :, EMBEDDING_SIZE:])
        # )
        # self.entity_prior = distributions.normal.Normal(
        #     loc=torch.zeros_like(entity_batch[:, :, :EMBEDDING_SIZE]),
        #     scale=torch.ones_like(entity_batch[:, :, :EMBEDDING_SIZE])
        # )

        # print('batch shapes', entity_sampler.batch_shape, self.entity_prior.batch_shape)
        # print('event shapes', entity_sampler.event_shape, self.entity_prior.event_shape)
        global_bias = global_bias_sampler.rsample((N_VARIATIONAL_SAMPLES,))
        biases = bias_sampler.rsample((N_VARIATIONAL_SAMPLES,))#.reshape(
            # N_VARIATIONAL_SAMPLES, -1, 2)
        entities = entity_sampler.rsample((N_VARIATIONAL_SAMPLES,))#.reshape(
            # N_VARIATIONAL_SAMPLES, -1, 2, EMBEDDING_SIZE)  # N_VAR_SAMPLES x BATCH_SIZE x 2 (user, item) x EMBEDDING_SIZE
        # print('hola', biases.shape, entities.shape)
        sum_users_items_biases = biases[:, entity_pos].sum(axis=2).mean(axis=0).squeeze()
        users_items_emb = entities[:, entity_pos].prod(axis=2).sum(axis=2).mean(axis=0)
        # print('final', sum_users_items_biases.shape, users_items_emb.shape)

        if self.mean_saved_mean_biases is not None:
            last_global_bias = self.saved_global_biases[-1]
            last_bias_term = self.saved_mean_biases[-1][x].sum(axis=1).squeeze()
            last_embed_term = self.saved_mean_entities[-1][x].prod(axis=1).sum(axis=1)

            mean_global_bias = self.mean_saved_global_biases
            mean_bias_term = self.mean_saved_mean_biases[x].sum(axis=1).squeeze()
            mean_embed_term = self.mean_saved_mean_entities[x].prod(axis=1).sum(axis=1)
            # print(self.mean_saved_mean_biases[x].shape, mean_bias_term.shape)
            # print(self.mean_saved_mean_entities[x].shape, mean_embed_term.shape)
            last_logits = last_global_bias + last_bias_term + last_embed_term
            mean_logits = mean_global_bias + mean_bias_term + mean_embed_term #  + 
        else:
            last_logits = None
            mean_logits = None

        std_dev = torch.sqrt(1 / LINK(self.alpha))
        unscaled_pred = global_bias + sum_users_items_biases + users_items_emb

        if self.output == 'reg':
            likelihood = distributions.normal.Normal(unscaled_pred, std_dev)
        else:
            likelihood = distributions.bernoulli.Bernoulli(logits=unscaled_pred)
        # print('global bias sampler', global_bias_sampler)
        # print('global bias prior', self.global_bias_prior)
        # print('bias sampler', bias_sampler)
        # print('bias prior', self.bias_prior)
        # print('entity sampler', entity_sampler)
        # print('entity prior', self.entity_prior)
        # a = distributions.normal.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        # b = distributions.normal.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        # print('oh hey', distributions.kl.kl_divergence(a, b))
        # print('oh hey', distributions.kl.kl_divergence(entity_sampler, entity_sampler))
        # print('oh hiya', distributions.kl.kl_divergence(entity_sampler, self.entity_prior))
        # print('oh hey', distributions.kl.kl_divergence(self.entity_prior, self.entity_prior))

        # print(
        #     distributions.kl.kl_divergence(global_bias_sampler, self.global_bias_prior).shape,
        #     distributions.kl.kl_divergence(bias_sampler, self.bias_prior).sum(axis=1).shape,
        #     distributions.kl.kl_divergence(entity_sampler, self.entity_prior).sum(axis=[1, 2]).shape#.sum(axis=3).shape,#.sum(axis=2).shape
        # )

        kl_bias = distributions.kl.kl_divergence(bias_sampler, self.bias_prior)
        # print('kl bias', kl_bias.shape)
        # print('bias sampler', bias_sampler)
        # print('entity sampler', entity_sampler)
        # print('entity prior', self.entity_prior)
        kl_entity = distributions.kl.kl_divergence(entity_sampler, self.entity_prior).sum(axis=1)
        # print('kl entity', kl_entity.shape)

        nb_occ_in_train = nb_occ[uniq_entities]
        nb_occ_user_in_train = nb_occ[uniq_users]
        nb_occ_item_in_train = nb_occ[uniq_items]
        # nb_occ_batch = torch.bincount(x.flatten())
        # print('nboccs', nb_occ_in_batch.shape, nb_occ_in_train.shape)
        # nb_occ_batch[x]

        user_normalizer = (nb_occ_user_in_batch / nb_occ_user_in_train).sum(axis=0)
        item_normalizer = (nb_occ_item_in_batch / nb_occ_item_in_train).sum(axis=0)
        # print('normalizers', user_normalizer.shape, item_normalizer.shape)

        # print('begin', ((kl_bias + kl_entity) * (nb_occ_in_batch / nb_occ_in_train)).shape)
        # print('ent', x)
        # print('ent', x <= N)
        # print('ent', (x <= N) * N)

        kl_rescaled = (
            (kl_bias + kl_entity) * (nb_occ_in_batch / nb_occ_in_train) *
            ((uniq_entities <= N) * N / user_normalizer + (uniq_entities > N) * M / item_normalizer)
        ).sum(axis=0)
        # print('rescaled', kl_rescaled.shape)

        return (likelihood,
            last_logits, mean_logits,
            distributions.kl.kl_divergence(global_bias_sampler, self.global_bias_prior) +
            kl_rescaled
        )


# def closure():
#     optimizer.zero_grad()
#     outputs, kl_bias, kl_entity = model(indices)
#     obj = -outputs.log_prob(target.float()).mean() + kl_bias.mean() + kl_entity.mean()
#     obj.backward()
#     return obj


# from torchcontrib.optim import SWA

model = CF(EMBEDDING_SIZE, output=OUTPUT_TYPE)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-4)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=LEARNING_RATE, history_size=10, max_iter=4, line_search_fn='strong_wolfe')
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

train_rmse = train_auc = train_map = 0.
losses = []
all_preds = []
for epoch in tqdm(range(N_EPOCHS)):
    losses = []
    pred = []
    truth = []
    for i, (indices, target) in enumerate(train_iter):
        # print('=' * 10, i)
        outputs, _, _, kl_term = model(indices)#.squeeze()
        # print(outputs)
        # print('indices', indices.shape, 'target', target.shape, outputs, 'ypred', len(y_pred), 'kl', kl_term.shape)
        # loss = loss_function(outputs, target)
        # print('kl', kl_bias.shape, kl_entity.shape)
        # print(outputs.sample()[:5], target[:5])
        loss = -outputs.log_prob(target.float()).mean() * nb_train_samples + kl_term
        # print('loss', loss)
        train_auc = -1

        y_pred = outputs.mean.squeeze().detach().numpy().tolist()
        losses.append(loss.item())
        pred.extend(y_pred)
        truth.extend(target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('preds', len(y_pred))
        # print('but target', target.shape)
        # print(len(pred), len(truth))
    # optimizer.swap_swa_sgd()

    # End of epoch
    if OUTPUT_TYPE == 'reg':
        pred = np.clip(pred, 1, 5)
        model.save_weights()
        train_rmse = mean_squared_error(truth, pred) ** 0.5
    else:
        train_auc = roc_auc_score(truth, pred)
        train_map = average_precision_score(truth, pred)

    '''print('test', outputs.sample()[:5], target[:5], loss.item())
        print('variance', torch.sqrt(1 / model.alpha))
        print('bias max abs', model.bias_params.weight.abs().max())
        print('entity max abs', model.entity_params.weight.abs().max())'''

    if epoch % DISPLAY_EPOCH_EVERY == 0:
        print('train pred', np.round(pred[:5], 4), truth[:5])
        print(f"Epoch {epoch}: Elbo {np.mean(losses):.4f} " +
              (f"Minibatch train RMSE {train_rmse:.4f}" if OUTPUT_TYPE == 'reg' else
               f"Minibatch train AUC {train_auc:.4f} " +
               f"Minibatch train MAP {train_map:.4f}"))

        print('precision', model.alpha, 'std dev', torch.sqrt(1 / nn.functional.softplus(model.alpha)))
        # print('bias max abs', model.bias_params.weight.abs().max())
        # print('entity max abs', model.entity_params.weight.abs().max())

        outputs, y_pred_of_last, y_pred_of_mean, _ = model(X_test)
        y_pred = outputs.mean.squeeze().detach().numpy()
        if OUTPUT_TYPE == 'reg':
            y_pred = y_pred.clip(1, 5)
            y_pred_of_mean = y_pred_of_mean.clip(1, 5)
            all_preds.append(y_pred.tolist())
            mean_pred = np.array(all_preds).mean(axis=0)
            print('test pred', np.round(y_pred[-5:], 4), y_test[-5:])    
            test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
            print('Test RMSE', test_rmse)
            test_rmse_all = mean_squared_error(y_test, mean_pred) ** 0.5
            print('Test RMSE all', test_rmse_all)
            test_rmse_of_last = mean_squared_error(y_test, y_pred_of_last) ** 0.5
            print('Test RMSE of last mean', test_rmse_of_last)
            test_rmse_of_mean = mean_squared_error(y_test, y_pred_of_mean) ** 0.5
            print('Test RMSE of mean', test_rmse_of_mean)
            # print('sanity check', y_pred[:5], mean_pred[:5], y_pred_of_mean[:5])
        else:
            test_auc = roc_auc_score(y_test, y_pred)
            test_map = average_precision_score(y_test, y_pred)            
            print(f'Test AUC {test_auc:.4f} Test MAP {test_map:.4f}')


if DATA == 'movielens':
    writer = SummaryWriter(log_dir='logs/embeddings')  # TBoard
    item_embeddings = list(model.parameters())[1][N:]
    user_meta = pd.DataFrame(np.arange(N), columns=('item',))
    user_meta['title'] = ''
    item_meta = df.sort_values('item')[['item', 'title']].drop_duplicates()
    metadata = pd.concat((user_meta, item_meta), axis=0)
    writer.add_embedding(
        item_embeddings, metadata=item_meta.values.tolist(),
        metadata_header=item_meta.columns.tolist())
    writer.close()
