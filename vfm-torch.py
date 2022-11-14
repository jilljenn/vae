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


LEARNING_RATE = 0.1
EMBEDDING_SIZE = 20
N_VARIATIONAL_SAMPLES = 1
OUTPUT_TYPE = 'reg'


# DATA = 'toy'
# DATA = 'movielens'
# DATA = 'movie100k'
DATA = 'movie100k'
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
    if DATA == 'movielens':
        N_EPOCHS = 50
        DISPLAY_EPOCH_EVERY = 2
        BATCH_SIZE = 1000
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
    elif DATA.startswith('movie100k'):
        N_EPOCHS = 50
        DISPLAY_EPOCH_EVERY = 2
        BATCH_SIZE = 800
        df = pd.read_csv('data/movie100k/data.csv')#.head(1000)
        if DATA.endswith('batch'):
            N_EPOCHS = 100
            DISPLAY_EPOCH_EVERY = 10
            df = df.head(1000)
        # films = pd.read_csv('ml-latest-small/movies.csv')
        # df = df.merge(films, on='movieId')
        df['user'] = np.unique(df['user'], return_inverse=True)[1]
        df['item'] = np.unique(df['item'], return_inverse=True)[1]
        N = df['user'].nunique()
        M = df['item'].nunique()
        # print(N, M, df.min(), df.max())
        # df['user'] -= 1
        df['item'] += N # - 1
        X = torch.LongTensor(df[['user', 'item']].to_numpy())
        y = torch.Tensor(df['rating'])
    elif DATA == 'movie100':
        N_EPOCHS = 100
        DISPLAY_EPOCH_EVERY = 5
        BATCH_SIZE = 100
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE) # , shuffle=True


class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size, output='reg'):
        super().__init__()
        self.output = output
        self.alpha = nn.Parameter(torch.Tensor([1e9]), requires_grad=True)
        self.global_bias = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        # self.alpha = torch.Tensor([1.])
        # nn.init.uniform_(self.alpha)
        self.bias_params = nn.Embedding(N + M, 2)  # w
        self.entity_params = nn.Embedding(N + M, 2 * embedding_size)  # V
        self.bias_prior = distributions.normal.Normal(
            torch.Tensor([0.]), torch.Tensor([1.]))
        self.entity_prior = distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(embedding_size), scale_tril=torch.eye(embedding_size))

    def forward(self, x):
        bias_batch = self.bias_params(x).reshape(-1, 2)
        entity_batch = self.entity_params(x).reshape(-1, 2 * EMBEDDING_SIZE)
        # print('first', bias_batch.shape, entity_batch.shape)
        scale_bias = nn.functional.softplus(bias_batch[:, 1])
        # scale_bias = torch.ones_like(scale_bias) * 1e-6
        bias_sampler = distributions.normal.Normal(
            bias_batch[:, 0], scale_bias)
        diag_scale_entity = nn.functional.softplus(entity_batch[:, EMBEDDING_SIZE:])
        # diag_scale_entity = torch.ones_like(diag_scale_entity) * 1e-6
        scale_entity = torch.diag_embed(diag_scale_entity)
        # print('scale entity', entity_batch.shape, scale_entity.shape)
        entity_sampler = distributions.multivariate_normal.MultivariateNormal(
            loc=entity_batch[:, :EMBEDDING_SIZE],
            scale_tril=scale_entity)
        # print('batch shapes', entity_sampler.batch_shape, self.entity_prior.batch_shape)
        # print('event shapes', entity_sampler.event_shape, self.entity_prior.event_shape)
        biases = bias_sampler.rsample((N_VARIATIONAL_SAMPLES,)).reshape(
            N_VARIATIONAL_SAMPLES, -1, 2)
        entities = entity_sampler.rsample((N_VARIATIONAL_SAMPLES,)).reshape(
            N_VARIATIONAL_SAMPLES, -1, 2, EMBEDDING_SIZE)  # N_VAR_SAMPLES x BATCH_SIZE x 2 (user, item) x EMBEDDING_SIZE
        # print('hola', biases.shape, entities.shape)
        sum_users_items_biases = biases.sum(axis=2).mean(axis=0).squeeze()
        users_items_emb = entities.prod(axis=2).sum(axis=2).mean(axis=0)
        # print('final', sum_users_items_biases.shape, users_items_emb.shape)
        std_dev = torch.sqrt(1 / nn.functional.softplus(self.alpha))
        unscaled_pred = self.global_bias + sum_users_items_biases + users_items_emb
        if self.output == 'reg':
            likelihood = distributions.normal.Normal(unscaled_pred, std_dev)
        else:
            likelihood = distributions.bernoulli.Bernoulli(logits=unscaled_pred)
        return (likelihood,
            distributions.kl.kl_divergence(bias_sampler, self.bias_prior),
            distributions.kl.kl_divergence(entity_sampler, self.entity_prior))


model = CF(EMBEDDING_SIZE, output=OUTPUT_TYPE)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
train_rmse = train_auc = train_map = 0.
losses = []
for epoch in tqdm(range(N_EPOCHS)):
    losses = []
    pred = []
    truth = []
    for indices, target in train_iter:
        # with torch.autograd.detect_anomaly():
        outputs, kl_bias, kl_entity = model(indices)#.squeeze()
        # print(outputs.shape)
        # loss = loss_function(outputs, target)
        # print('kl', kl_bias.shape, kl_entity.shape)
        # print(outputs.sample()[:5], target[:5])
        loss = -outputs.log_prob(target.float()).mean() #+ kl_bias.mean() + kl_entity.mean()
        train_auc = -1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = outputs.mean.detach().numpy().clip(1, 5)
        pred.extend(y_pred)
        truth.extend(target)
        losses.append(loss.item())

    # End of epoch
    if OUTPUT_TYPE == 'reg':
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

        # print('precision', model.alpha, 'std dev', torch.sqrt(1 / model.alpha))
        # print('bias max abs', model.bias_params.weight.abs().max())
        # print('entity max abs', model.entity_params.weight.abs().max())

        outputs, _, _ = model(X_test)
        y_pred = outputs.mean.detach().numpy().clip(1, 5)
        print('test pred', np.round(y_pred[-5:], 4), y_test[-5:])    
        if OUTPUT_TYPE == 'reg':
            test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
            print('Test RMSE', test_rmse)
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
