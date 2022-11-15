"""
Matrix completion on toy and Movielens datasets
JJV for Deep Learning course, 2022
"""
import torch
from torch import nn, distributions
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, average_precision_score
import pandas as pd
from prepare import load_data
import matplotlib.pyplot as plt
from torchmin import minimize

# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)
device = torch.device('cuda')  # cuda

def draw_graph(start, watch=[]):
    from graphviz import Digraph
    node_attr = dict(style='filled',
                    shape='box',
                    align='left',
                    fontsize='12',
                    ranksep='0.1',
                    height='0.2')
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    assert(hasattr(start, "grad_fn"))
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)
    size_per_element = 0.15
    min_size = 12
    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename='net_graph.jpg')
def _draw_graph(var, graph, watch=[], seen=[], indent="", pobj=None):
    ''' recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing.'''
    from rich import print
    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = str(type(joy)).replace(
                        "class", "").replace("'", "").replace(" ", "")
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)
                    if hasattr(joy, 'variable'):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"
                            for (name, obj) in watch:
                                if obj is happy:
                                    label += " \U000023E9 " + \
                                        "[b][u][color=#FF00FF]" + name + \
                                        "[/color][/u][/b]"
                                    label_graph += name
                                    colour_graph = "blue"
                                    break
                                vv = [str(obj.shape[x])
                                    for x in range(len(obj.shape))]
                                label += " [["
                                label += ', '.join(vv)
                                label += "]]"
                            label += " " + str(happy.var())
                    graph.node(str(joy), label_graph, fillcolor=colour_graph)
                    print(indent + label)
                    _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                    if pobj is not None:
                        graph.edge(str(pobj), str(joy))

LEARNING_RATE = 0.1
EMBEDDING_SIZE = 20
N_VARIATIONAL_SAMPLES = 1
OUTPUT_TYPE = 'reg'
LBFGS = False

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
    X_train = None
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
        DISPLAY_EPOCH_EVERY = 1
        BATCH_SIZE = 1024
        # BATCH_SIZE = 100000
        '''df = pd.read_csv('data/movie100k/data.csv')#.head(1000)
                                if DATA.endswith('batch'):
                                    N_EPOCHS = 100
                                    DISPLAY_EPOCH_EVERY = 10
                                    df = df.head(1000)'''
        # films = pd.read_csv('ml-latest-small/movies.csv')
        # df = df.merge(films, on='movieId')

        N, M, X_train, X_test, y_train, y_test = load_data('movie100k')
        nb_samples_train = len(X_train)
        X_train = torch.LongTensor(X_train)
        y_train = torch.Tensor(y_train)
        X_test = torch.LongTensor(X_test)
        y_test = torch.Tensor(y_test)

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

    if X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test, y_test_cpu = y_test.to(device), y_test
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE) # , shuffle=True

all_entities = torch.arange(start=0, end=N+M, device=device)

class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size, output='reg'):
        super().__init__()
        self.output = output
        self.alpha = nn.Parameter(torch.Tensor([1e3]), requires_grad=True)
        self.global_bias_mean = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        self.global_bias_scale = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_global_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_item_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_entity_prior = nn.Parameter(torch.ones((1, EMBEDDING_SIZE)), requires_grad=True)
        self.prec_item_entity_prior = nn.Parameter(torch.ones((1, EMBEDDING_SIZE)), requires_grad=True)
        # self.alpha = torch.Tensor([1.])
        # nn.init.uniform_(self.alpha)
        self.bias_params = nn.Embedding(N + M, 2)  # w
        self.entity_params = nn.Embedding(N + M, 2 * embedding_size)  # V
        self.drawn = False

    def update_priors(self):

        # print('kikoo entity prior',
        #     torch.nn.functional.softplus(torch.cat((
        #             self.prec_user_entity_prior.repeat(N, 1),
        #             self.prec_item_entity_prior.repeat(M, 1)
        #         ))).shape)

        self.global_bias_prior = distributions.normal.Normal(
            torch.Tensor([0.]).to(device),
            torch.nn.functional.softplus(self.prec_global_bias_prior))
        self.bias_prior = distributions.normal.Normal(
            torch.zeros(N + M).to(device),
            torch.nn.functional.softplus(torch.cat((
                self.prec_user_bias_prior.repeat(N),
                self.prec_item_bias_prior.repeat(M)
            )))
            # torch.ones(N + M).to(device)
        )
        self.entity_prior = distributions.independent.Independent(
            distributions.normal.Normal(
                torch.zeros((N + M, EMBEDDING_SIZE), device=device),
                torch.nn.functional.softplus(torch.cat((
                    self.prec_user_entity_prior.repeat(N, 1),
                    self.prec_item_entity_prior.repeat(M, 1)
                )))
            ), 1
        )

    def draw(self):
        if self.drawn:
            return
        # Global bias
        self.global_bias_sampler = distributions.normal.Normal(
            self.global_bias_mean,
            nn.functional.softplus(self.global_bias_scale)
        )
        # Biases and entities
        all_bias_params = self.bias_params(all_entities)
        self.bias_sampler = distributions.normal.Normal(
            all_bias_params[:, 0],
            nn.functional.softplus(all_bias_params[:, 1])
        )
        all_entity_params = self.entity_params(all_entities)
        self.entity_sampler = distributions.independent.Independent(
            distributions.normal.Normal(
                all_entity_params[:, :EMBEDDING_SIZE],
                nn.functional.softplus(all_entity_params[:, EMBEDDING_SIZE:]),
            ), 1
        )

        self.global_bias = self.global_bias_sampler.rsample((N_VARIATIONAL_SAMPLES,)).mean()  # Only one scalar
        self.all_bias = self.bias_sampler.rsample((N_VARIATIONAL_SAMPLES,))
        self.all_entity = self.entity_sampler.rsample((N_VARIATIONAL_SAMPLES,))
        # self.drawn = True

    def forward(self, x):
        self.update_priors()
        self.draw()
        biases = torch.index_select(self.all_bias, 1, x.flatten()).reshape((N_VARIATIONAL_SAMPLES, -1, 2))
        entities = torch.index_select(self.all_entity, 1, x.flatten()).reshape((N_VARIATIONAL_SAMPLES, -1, 2, EMBEDDING_SIZE))

        # print('hola', biases.shape, entities.shape)
        sum_users_items_biases = biases.sum(axis=2).mean(axis=0).squeeze()
        users_items_emb = entities.prod(axis=2).sum(axis=2).mean(axis=0)
        # print('final', sum_users_items_biases.shape, users_items_emb.shape)
        std_dev = torch.sqrt(1 / nn.functional.softplus(self.alpha))
        # print('global bias, sum_users_items_biases', self.global_bias.shape, sum_users_items_biases.shape)
        unscaled_pred = self.global_bias + sum_users_items_biases + users_items_emb
        # print('unscaled pred', unscaled_pred.shape)
        if self.output == 'reg':
            likelihood = distributions.normal.Normal(unscaled_pred, std_dev)
        else:
            likelihood = distributions.bernoulli.Bernoulli(logits=unscaled_pred)
        return (likelihood,
            [
                distributions.kl.kl_divergence(self.global_bias_sampler, self.global_bias_prior),
                distributions.kl.kl_divergence(self.bias_sampler, self.bias_prior),
                distributions.kl.kl_divergence(self.entity_sampler, self.entity_prior)
            ])

model = CF(EMBEDDING_SIZE, output=OUTPUT_TYPE).to(device)
mse_loss = nn.MSELoss()
if LBFGS:
    if True:
        result = minimize(fn, x0, 'BFGS')
    else:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=LEARNING_RATE, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            outputs, kl_bias, kl_entity = model(indices)
            obj = -outputs.log_prob(target.float()).mean() + kl_bias.mean() + kl_entity.mean()
            obj.backward()
            return obj
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
train_rmse = train_auc = train_map = 0.
losses = []
y_pred_all = np.zeros(len(test_dataset))
rmses = {'all': [], 'this': []}

with Progress(
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn('[progress.description]Elbo: {task.fields[elbo]:.4f} RMSE: {task.fields[test_rmse]:.4f} All: {task.fields[all_rmse]:.4f}'),
) as progress:
  training = progress.add_task('VFM training...', total=N_EPOCHS, elbo=float('nan'), test_rmse=float('nan'), all_rmse=float('nan'))
  for epoch in range(N_EPOCHS):
    model.drawn = False

    losses = []
    pred = []
    truth = []
    # print('drawn', model.drawn)
    for indices, target in train_iter:
        # with torch.autograd.detect_anomaly():
        outputs, kls = model(indices)#.squeeze()
        # print(outputs.shape)
        # loss = loss_function(outputs, target)
        # print('KLs', [t.shape for t in kls])
        # print(outputs.sample()[:5], target[:5])
        loss = -outputs.log_prob(target.float()).mean() * nb_samples_train + sum([kl.sum() for kl in kls])
        train_auc = -1

        y_pred = outputs.sample().detach().cpu().numpy().clip(1, 5)

        if not LBFGS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(closure)

        # print(y_pred.shape)
        pred.extend(y_pred)
        truth.extend(target.cpu())
        losses.append(loss.item())

    # End of epoch
    if OUTPUT_TYPE == 'reg':
        # print('truth pred', len(truth), len(pred), truth[:5], pred[:5])
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

        outputs, _ = model(X_test)
        y_pred = outputs.sample().detach().cpu().numpy().clip(1, 5)
        if epoch >= 5:
            y_pred_all += y_pred
        print('test pred')
        for name, pred in zip(['this', 'all ', 'true'],
                [y_pred[-5:], y_pred_all[-5:]/max(1, epoch-4), y_test_cpu[-5:].numpy()]):
            print(name, ' '.join(f'{i:.4f}' for i in pred))
        if OUTPUT_TYPE == 'reg':
            test_rmse = mean_squared_error(y_test_cpu, y_pred) ** 0.5
            all_rmse = mean_squared_error(y_test_cpu, y_pred_all / max(1, (epoch - 4))) ** 0.5
            rmses['this'].append(test_rmse)
            rmses['all'].append(all_rmse)
            print('Test RMSE', test_rmse, 'All RMSE', all_rmse)
        else:
            test_auc = roc_auc_score(y_test_cpu, y_pred)
            test_map = average_precision_score(y_test_cpu, y_pred)
            print(f'Test AUC {test_auc:.4f} Test MAP {test_map:.4f}')
    progress.update(training, advance=1, elbo=np.mean(losses), test_rmse=test_rmse, all_rmse=all_rmse)
    print(model.alpha)

plt.plot(rmses['this'], label='VFM this')
plt.plot(rmses['all'], label='VFM all')
plt.legend()
plt.savefig('VFMrmses.png')

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
