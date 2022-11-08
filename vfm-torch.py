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
import pandas as pd


LEARNING_RATE = 0.1
EMBEDDING_SIZE = 1
N_VARIATIONAL_SAMPLES = 1


# DATA = 'toy'
# DATA = 'movielens'
DATA = 'movie100k'
if DATA == 'toy':
    N_EPOCHS = 1000
    DISPLAY_EPOCH_EVERY = 100
    BATCH_SIZE = 50
    N, K, M = 10, 3, 5
    U = np.random.normal(size=(N, K))
    V = np.random.normal(size=(M, K))
    R = U @ V.T
    X = []
    y = []
    for i in range(N):  # Can be done using pd.unstack
        for j in range(M):
            X.append((i, N + j))
            y.append(R[i, j])
elif DATA == 'movielens':
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
else:
    N_EPOCHS = 50
    DISPLAY_EPOCH_EVERY = 2
    BATCH_SIZE = 1000
    df = pd.read_csv('data/movie100k/data.csv').head(1000)
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

X = torch.LongTensor(X)
y = torch.Tensor(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential(
    nn.Embedding(N + M, EMBEDDING_SIZE),
    nn.Flatten(),
    nn.Linear(2 * EMBEDDING_SIZE, 1),
)

class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([1.]))
        # self.alpha = torch.Tensor([1.])
        nn.init.uniform_(self.alpha)
        self.bias_params = nn.Embedding(N + M, 2)
        self.entity_params = nn.Embedding(N + M, 2 * embedding_size)
        self.bias_prior = distributions.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))

    def forward(self, x):
        bias_batch = self.bias_params(x).reshape(-1, 2)
        entity_batch = self.entity_params(x).reshape(-1, 2 * EMBEDDING_SIZE)
        # print('first', bias_batch.shape, entity_batch.shape)
        scale_bias = nn.functional.softplus(bias_batch[:, 1])
        bias_sampler = distributions.normal.Normal(
            bias_batch[:, 0], scale_bias)
        scale_entity = torch.diag_embed(
            nn.functional.softplus(entity_batch[:, EMBEDDING_SIZE:]))
        # print('scale entity', entity_batch.shape, scale_entity.shape)
        entity_sampler = distributions.multivariate_normal.MultivariateNormal(
            loc=entity_batch[:, :EMBEDDING_SIZE],
            scale_tril=scale_entity)
        biases = bias_sampler.rsample((N_VARIATIONAL_SAMPLES,)).reshape(
            N_VARIATIONAL_SAMPLES, -1, 2)
        entities = entity_sampler.rsample((N_VARIATIONAL_SAMPLES,)).reshape(
            N_VARIATIONAL_SAMPLES, -1, 2, EMBEDDING_SIZE)
        # print('hola', biases.shape, entities.shape)
        sum_users_items_biases = biases.sum(axis=2).mean(axis=0).squeeze()
        users_items_emb = entities.prod(axis=2).sum(axis=2).mean(axis=0)
        # print('final', sum_users_items_biases.shape, users_items_emb.shape)
        std_dev = torch.sqrt(1 / nn.functional.softplus(self.alpha))
        return (distributions.normal.Normal(
            sum_users_items_biases + users_items_emb, std_dev),
            distributions.kl.kl_divergence(bias_sampler, self.bias_prior))


model = CF(EMBEDDING_SIZE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []
for epoch in tqdm(range(N_EPOCHS)):
    losses = []
    for indices, target in train_iter:
        # with torch.autograd.detect_anomaly():
        outputs, kl_term = model(indices)#.squeeze()
        # print(outputs.shape)
        # loss = loss_function(outputs, target)
        # print('kl', kl_term.shape)
        loss = -outputs.log_prob(target).mean() + kl_term.mean()

        '''print('test', outputs.sample()[:5], target[:5], loss.item())
            print('variance', torch.sqrt(1 / model.alpha))
            print('bias max abs', model.bias_params.weight.abs().max())
            print('entity max abs', model.entity_params.weight.abs().max())'''

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if epoch % DISPLAY_EPOCH_EVERY == 0:
        print(f"Epoch {epoch}: Train MSE {np.mean(losses)}")

        # print('precision', model.alpha, 'std dev', torch.sqrt(1 / model.alpha))
        # print('bias max abs', model.bias_params.weight.abs().max())
        # print('entity max abs', model.entity_params.weight.abs().max())

        outputs, _ = model(X_test)
        y_pred = outputs.sample()
        test_loss = loss_function(y_pred, y_test)
        print('pred', y_pred[:5], y_test[:5])
        print('Test MSE', test_loss)


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
