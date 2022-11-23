"""
Matrix completion on toy and Movielens datasets
JJV for Deep Learning course, 2022
"""
import numpy as np

import torch
from torch import nn, distributions
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, average_precision_score
import pandas as pd
from prepare import load_data
import matplotlib.pyplot as plt
from torchmin import Minimizer

# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)
device = torch.device('cpu')  # cuda

# DATA = 'toy'
# DATA = 'movielens'
# DATA = 'movie100k'
DATA = 'movie100k'
# DATA = 'movie1M'
# DATA = 'movie10M'
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
        BATCH_SIZE = 400
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
        BATCH_SIZE = 800
        BATCH_SIZE = 8000
        # BATCH_SIZE = 100000
        N_GROUPS = 2
        '''df = pd.read_csv('data/movie100k/data.csv')#.head(1000)
                                if DATA.endswith('batch'):
                                    N_EPOCHS = 100
                                    DISPLAY_EPOCH_EVERY = 10
                                    df = df.head(1000)'''
        # films = pd.read_csv('ml-latest-small/movies.csv')
        # df = df.merge(films, on='movieId')

        N, M, X_train, X_test, y_train, y_test, _ = load_data('movie100k')
        nb_samples_train = len(X_train)
        X_train = torch.LongTensor(X_train)
        y_train = torch.Tensor(y_train)
        X_test = torch.LongTensor(X_test)
        y_test = torch.Tensor(y_test)
    elif DATA.startswith('movie1M'):
        N_EPOCHS = 200
        DISPLAY_EPOCH_EVERY = 1
        BATCH_SIZE = 800
        BATCH_SIZE = 80000
        # BATCH_SIZE = 100000

        N, M, X_train, X_test, y_train, y_test, _ = load_data('movie1M')
        nb_samples_train = len(X_train)
        X_train = torch.LongTensor(X_train)
        y_train = torch.Tensor(y_train)
        X_test = torch.LongTensor(X_test)
        y_test = torch.Tensor(y_test)
    elif DATA.startswith('movie10M'):
        N_EPOCHS = 10
        DISPLAY_EPOCH_EVERY = 1
        BATCH_SIZE = 800
        BATCH_SIZE = 800000
        # BATCH_SIZE = 100000

        N, M, X_train, X_test, y_train, y_test, _ = load_data('movie10M')
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
entity_count = torch.bincount(X_train.flatten()).float()

eps = 1e-9

class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size=2, n_groups=2, group_sizes=None,
            n_var_samples=1, alpha_0=300, output='reg'):
        super().__init__()
        self.output = output
        self.alpha = nn.Parameter(torch.Tensor([alpha_0]))
        self.drawn = False
        self.n_var_samples = n_var_samples
        self.embedding_size = embedding_size
        self.n_groups = n_groups
        self.group_sizes = [N, M] if group_sizes is None else group_sizes
        assert n_groups == len(self.group_sizes)
        self.link = torch.abs

        start_scale = 0.2
        var_prior_mean = True

        # Global Bias
        self.mean_global_bias_prior  = nn.Parameter(
            torch.Tensor([0.]),
            requires_grad=var_prior_mean
        )
        self.scale_global_bias_prior = nn.Parameter(torch.Tensor([1.]))

        self.mean_global_bias  = nn.Parameter(
            torch.normal(torch.zeros(1), torch.ones(1))
        )
        self.scale_global_bias = nn.Parameter(torch.Tensor([start_scale]))

        # Entity W params
        self.mean_group_bias_prior  = nn.ParameterList([
            nn.Parameter(torch.zeros(1), requires_grad=var_prior_mean)
            for _ in range(n_groups)
        ])
        self.scale_group_bias_prior = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in range(n_groups)
        ])

        # self.bias_params = nn.Embedding(N + M, 2)  # w
        self.bias_params = nn.Parameter(torch.cat([
            torch.cat((
                torch.normal(torch.zeros(ni, 1), 1e-1 * torch.ones(ni, 1)),
                start_scale * torch.ones(ni, 1)
            ), axis=1)
            for ni in self.group_sizes
        ]))

        # Entity V params
        self.mean_group_entity_prior  = nn.ParameterList([
            nn.Parameter(
                torch.zeros(embedding_size),
                requires_grad=var_prior_mean
            )
            for _ in range(n_groups)
        ])
        self.scale_group_entity_prior = nn.ParameterList([
            nn.Parameter(torch.ones(embedding_size))
            for _ in range(n_groups)
        ])

        # self.entity_params = nn.Embedding(N + M, 2 * embedding_size)  # V
        self.entity_params = nn.Parameter(torch.cat([
            torch.cat((
                torch.normal(
                    torch.zeros(ni, embedding_size),
                    1e-7 * torch.ones(ni, embedding_size)
                ),
                start_scale * torch.ones(ni, embedding_size)
            ), axis=1)
            for ni in self.group_sizes
        ]))

    def update_priors(self, indices):
        group_sizes = list(map(len, indices))

        self.global_bias_prior = distributions.normal.Normal(
            self.mean_global_bias_prior,
            self.link(self.scale_global_bias_prior)
        )

        self.bias_prior = distributions.normal.Normal(
            torch.cat([
                self.mean_group_bias_prior[i_group].repeat(group_sizes[i_group])
                for i_group in range(self.n_groups)
            ]),
            self.link(torch.cat([
                self.scale_group_bias_prior[i_group].repeat(group_sizes[i_group])
                for i_group in range(self.n_groups)
            ]))
        )

        self.entity_prior = distributions.normal.Normal(
            torch.cat([
                self.mean_group_entity_prior[i_group].repeat(group_sizes[i_group], 1)
                for i_group in range(self.n_groups)
            ]),
            self.link(torch.cat([
                self.scale_group_entity_prior[i_group].repeat(group_sizes[i_group], 1)
                for i_group in range(self.n_groups)
            ]))
        )

    def draw(self, indices):
        if self.drawn:
            return

        # Global bias
        self.global_bias_sampler = distributions.normal.Normal(
            self.mean_global_bias,
            self.link(self.scale_global_bias)
        )

        # Biases and entities
        bias_params = self.bias_params[torch.cat(indices)]
        self.bias_sampler = distributions.normal.Normal(
            bias_params[:, 0],
            self.link(bias_params[:, 1])
        )

        entity_params = self.entity_params[torch.cat(indices)]
        self.entity_sampler = distributions.normal.Normal(
            entity_params[:, :self.embedding_size],
            self.link(entity_params[:, self.embedding_size:])
        )

        # self.global_bias = self.global_bias_sampler \
        #                        .rsample((self.n_var_samples,)) \
        #                        .squeeze(dim=1)
        # self.all_bias = self.bias_sampler.rsample((self.n_var_samples,))
        # self.all_entity = self.entity_sampler.rsample((self.n_var_samples,))

        # self.drawn = True

    def forward(self, x, x_unique, closed_form_loss=False, target=False):
        self.update_priors(x_unique)
        self.draw(x_unique)

        batch_entities = [
            x[i] + sum(map(len, x_unique[:i]))
            for i in range(len(x))
        ]
        biases   = torch.cat([
            torch.index_select(self.bias_sampler.mean, 0, x_entities)
                 .reshape((-1, 1))
            for x_entities in batch_entities
        ], axis=1)
        entities = torch.cat([
            torch.index_select(self.entity_sampler.mean, 0, x_entities)
                 .reshape((-1, 1, self.embedding_size))
            for x_entities in batch_entities
        ], axis=1)

        sum_groups_biases = biases.sum(axis=1)
        groups_emb = entities.prod(axis=1).sum(axis=1)
        unscaled_pred = (
              self.global_bias_sampler.mean
            + sum_groups_biases
            + groups_emb
        )
        # sum_users_items_biases = biases.sum(axis=2)
        # users_items_emb = entities.prod(axis=2).sum(axis=2)
        # unscaled_pred = (
        #       self.global_bias.mean(axis=0)
        #     + sum_users_items_biases.mean(axis=0)
        #     + users_items_emb.mean(axis=0)
        # )

        if self.output == 'reg':
            std_dev = torch.sqrt(1 / self.link(self.alpha))
            likelihood = distributions.normal.Normal(unscaled_pred, std_dev)
        else:
            likelihood = distributions.bernoulli.Bernoulli(logits=unscaled_pred)

        kls = [
            distributions.kl.kl_divergence(self.global_bias_sampler, self.global_bias_prior),
            distributions.kl.kl_divergence(self.bias_sampler, self.bias_prior),
            distributions.kl.kl_divergence(self.entity_sampler, self.entity_prior)
        ]

        if closed_form_loss:
            y_n_bar = (
                # mu_0'
                + self.mean_global_bias
                # sum_i mu_w_i' x_n_i
                + sum([
                    self.bias_params[x_unique[i_group][x[i_group]], 0]
                    for i_group in range(self.n_groups)
                ])
                # sum_i sum_j x_n_i x_n_j + sum_k mu_v_i,k' mu_v_j,k'
                + sum([
                    torch.einsum(
                        'ab,ab->a',
                        self.entity_params[
                            x_unique[i_group][x[i_group]],
                            :self.embedding_size
                        ],
                        self.entity_params[
                            x_unique[j_group][x[j_group]],
                            :self.embedding_size
                        ]
                    )
                    for i_group in range(self.n_groups)
                    for j_group in range(i_group+1, self.n_groups)
                ])
            )
            T_n = (
                # sigma_0^2'
                + self.scale_global_bias ** 2
                # sum_i sigma_w_i' x_n_i^2
                + sum([
                    self.bias_params[x_unique[i_group][x[i_group]], 1] ** 2
                    for i_group in range(self.n_groups)
                ])
                # sum_i sum_j x_n_i^2 x_n_j^2
                # + sum_k mu_v_i,k'^2 sigma_v_j,k'
                + sum([
                    + torch.einsum(
                        'ab,ab->a',
                        self.entity_params[
                            x_unique[i_group][x[i_group]],
                            :self.embedding_size
                        ] ** 2,
                        self.entity_params[
                            x_unique[j_group][x[j_group]],
                            self.embedding_size:
                        ] ** 2
                    )
                    # +       mu_v_j,k'^2 sigma_v_i,k'
                    + torch.einsum(
                        'ab,ab->a',
                        self.entity_params[
                            x_unique[j_group][x[j_group]],
                            :self.embedding_size
                        ] ** 2,
                        self.entity_params[
                            x_unique[i_group][x[i_group]],
                            self.embedding_size:
                        ] ** 2
                    )
                    # +       sigma_v_i,k' sigma_v_j,k'
                    + torch.einsum(
                        'ab,ab->a',
                        self.entity_params[
                            x_unique[i_group][x[i_group]],
                            self.embedding_size:
                        ] ** 2,
                        self.entity_params[
                            x_unique[j_group][x[j_group]],
                            self.embedding_size:
                        ] ** 2
                    )
                    for i_group in range(self.n_groups)
                    for j_group in range(i_group + 1, self.n_groups)
                ])
            )
            partial_loss = (
                + 1/2 * self.link(self.alpha).log()
                - self.link(self.alpha) / 2
                    * ((target - y_n_bar)**2 + T_n)
            ).sum()

            return (likelihood, kls, partial_loss)

        return (likelihood, kls)

default_progress = {
    'n_elbo': 'Elbo',
    'n_rmse': 'RMSE',
    'n_all': 'All',
    'elbo': float('nan'),
    'test_rmse': float('nan'),
    'all_rmse': float('nan')
}

def run(lr=0.02, alpha_0=300, embedding_size=2, n_groups=2, group_sizes=None,
        n_var_samples=1):
    model = CF(
        embedding_size=embedding_size,
        n_groups=n_groups,
        group_sizes=group_sizes,
        n_var_samples=n_var_samples,
        alpha_0=alpha_0,
        output=OUTPUT_TYPE
    ).to(device)
    # mse_loss = nn.MSELoss()
    train_rmse = train_auc = train_map = 0.
    elbos = []
    y_preds = []
    y_pred_all = np.zeros(len(test_dataset))
    rmses = {'all': [], 'this': [], 'train': []}
    params = {
        'mmin': [[] for _ in range(n_groups)],
        'mmax': [[] for _ in range(n_groups)],
        'vmin': [[] for _ in range(n_groups)],
        'vmax': [[] for _ in range(n_groups)],
    }

    if LBFGS:
        lbfgs_iter = 100
        opt_task = progress.add_task(
            'Optimizing...',
            total=lbfgs_iter,
            **default_progress
        )
        if True:
            def cb(_):
                progress.update(opt_task, advance=1)
            optimizer = Minimizer(
                model.parameters(),
                method='l-bfgs',
                tol=1e-5,
                options={
                    'lr': lr,
                },
                max_iter=lbfgs_iter,
                disp=0,
                callback=cb
            )
        else:
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                line_search_fn='strong_wolfe'
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # , weight_decay=1e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(N_EPOCHS):
        losses = []
        pred = []
        truth = []

        for i_group in range(n_groups):
            print(model.entity_params[i_group][:embedding_size].min(), model.entity_params[i_group][:embedding_size].max())
            print(torch.abs(model.entity_params[i_group][embedding_size:]).min(), torch.abs(model.entity_params[i_group][embedding_size:]).max())
            params['mmin'][i_group].append(model.entity_params[i_group][:embedding_size].min().cpu().detach().numpy())
            params['mmax'][i_group].append(model.entity_params[i_group][:embedding_size].max().cpu().detach().numpy())
            params['vmin'][i_group].append(torch.abs(model.entity_params[i_group][embedding_size:]).min().cpu().detach().numpy())
            params['vmax'][i_group].append(torch.abs(model.entity_params[i_group][embedding_size:]).max().cpu().detach().numpy())

        progress.reset(batch_task)
        for indices, target in train_iter:
            user_present, inverse_user, batch_user_count = torch.unique(
                    indices[:,0],
                return_inverse=True,
                return_counts=True
            )
            item_present, inverse_item, batch_item_count = torch.unique(
                    indices[:,1],
                return_inverse=True,
                return_counts=True
            )

            if not LBFGS:
                outputs, kls, partial_loss = model(
                    (inverse_user, inverse_item),
                    (user_present, item_present),
                    closed_form_loss=True,
                    target=target
                )

                y_pred = outputs.mean.detach().cpu().numpy().clip(1, 5)

                loss = (
                    # - outputs.log_prob(target.float()).sum()
                    - partial_loss
                    + kls[0] / len(train_iter)
                    + (
                        (kls[1] + kls[2].sum(axis=1))
                        * torch.concat((
                            batch_user_count,
                            batch_item_count
                        ))
                        / entity_count[torch.concat((
                            user_present,
                            item_present
                        ))]
                    ).sum()
                )

                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                progress.reset(opt_task)

                def closure():
                    optimizer.zero_grad()
                    outputs, kls, partial_loss = model(
                        (inverse_user, inverse_item),
                        (user_present, item_present),
                        closed_form_loss=True,
                        target=target
                    )

                    loss = (
                        # - outputs.log_prob(target.float()).sum()
                        - partial_loss
                        + kls[0] / len(train_iter)
                        + (
                            (kls[1] + kls[2].sum(axis=1))
                            * torch.concat((
                                batch_user_count,
                                batch_item_count
                            ))
                            / entity_count[torch.concat((
                                user_present,
                                item_present
                            ))]
                        ).sum()
                    )
                    return loss
                optimizer.step(closure)

                outputs, _ = model(
                    (inverse_user, inverse_item),
                    (user_present, item_present),
                )
                y_pred = outputs.mean.detach().cpu().numpy().clip(1, 5)

            # print(y_pred.shape)
            pred.extend(y_pred)
            truth.extend(target.cpu())
            progress.update(batch_task, advance=1)

        # End of epoch
        if OUTPUT_TYPE == 'reg':
            # print('truth pred', len(truth), len(pred), truth[:5], pred[:5])
            train_rmse = mean_squared_error(truth, pred) ** 0.5
            rmses['train'].append(train_rmse)
        else:
            train_auc = roc_auc_score(truth, pred)
            train_map = average_precision_score(truth, pred)

        '''print('test', outputs.mean[:5], target[:5], loss.item())
            print('variance', torch.sqrt(1 / model.alpha))
            print('bias max abs', model.bias_params.weight.abs().max())
            print('entity max abs', model.entity_params.weight.abs().max())'''

        elbo = np.mean(losses) if len(losses) > 0 else float('nan')
        elbos.append(elbo)
        if epoch % DISPLAY_EPOCH_EVERY == 0:
            print('train pred', np.round(pred[:5], 4), truth[:5])
            print(f"Epoch {epoch}: Elbo {elbo:.4f} " +
                  (f"Minibatch train RMSE {train_rmse:.4f}" if OUTPUT_TYPE == 'reg' else
                   f"Minibatch train AUC {train_auc:.4f} " +
                   f"Minibatch train MAP {train_map:.4f}"))

            # print('precision', model.alpha, 'std dev', torch.sqrt(1 / model.alpha))
            # print('bias max abs', model.bias_params.weight.abs().max())
            # print('entity max abs', model.entity_params.weight.abs().max())

            user_present, inverse_user, batch_user_count = torch.unique(
                    X_test[:,0],
                return_inverse=True,
                return_counts=True
            )
            item_present, inverse_item, batch_item_count = torch.unique(
                    X_test[:,1],
                return_inverse=True,
                return_counts=True
            )
            outputs, _ = model(
                (inverse_user, inverse_item),
                (user_present, item_present)
            )
            y_pred = outputs.mean.detach().cpu().numpy().clip(1, 5)
            y_preds.append(y_pred)
            nb_all = 1
            if epoch >= 1000:
                y_pred_all = np.sum(y_preds[-10:], axis=0)
                nb_all = 10
            elif epoch >= 5:
                y_pred_all += y_pred
                nb_all = epoch - 4
            print('test pred')
            for name, pred in zip(['this', 'all ', 'true'],
                    [y_pred[-5:], y_pred_all[-5:]/nb_all, y_test_cpu[-5:].numpy()]):
                print(name, ' '.join(f'{i:.4f}' for i in pred))
            if OUTPUT_TYPE == 'reg':
                test_rmse = mean_squared_error(y_test_cpu, y_pred) ** 0.5
                all_rmse = mean_squared_error(y_test_cpu, y_pred_all / nb_all) ** 0.5
                rmses['this'].append(test_rmse)
                rmses['all'].append(all_rmse)
                print('Test RMSE', test_rmse, 'All RMSE', all_rmse)
            else:
                test_auc = roc_auc_score(y_test_cpu, y_pred)
                test_map = average_precision_score(y_test_cpu, y_pred)
                print(f'Test AUC {test_auc:.4f} Test MAP {test_map:.4f}')
        progress.update(training, advance=1, elbo=elbo, test_rmse=test_rmse, all_rmse=all_rmse)
        print(model.alpha)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 18))
    ax1.set_ylim([0.7, 1.4])
    ax1.plot(rmses['train'], label='VFM train')
    ax1.plot(rmses['this'], label='VFM this')
    ax1.plot(rmses['all'], label='VFM all')
    ax1.legend()
    ax2.plot(elbos, label='elbo')
    ax2.legend()
    for i_group in range(n_groups):
        ax3.plot(params['mmin'][i_group], label=f'{i_group} mean min')
        ax3.plot(params['mmax'][i_group], label=f'{i_group} mean max')
        ax3.plot(params['vmin'][i_group], label=f'{i_group} var min')
        ax3.plot(params['vmax'][i_group], label=f'{i_group} var max')
    ax3.legend()
    fig.savefig(f'VFMrmses_{all_rmse:.4f}.png')

    return test_rmse

# if DATA == 'movielens':
#     writer = SummaryWriter(log_dir='logs/embeddings')  # TBoard
#     item_embeddings = list(model.parameters())[1][N:]
#     user_meta = pd.DataFrame(np.arange(N), columns=('item',))
#     user_meta['title'] = ''
#     item_meta = df.sort_values('item')[['item', 'title']].drop_duplicates()
#     metadata = pd.concat((user_meta, item_meta), axis=0)
#     writer.add_embedding(
#         item_embeddings, metadata=item_meta.values.tolist(),
#         metadata_header=item_meta.columns.tolist())
#     writer.close()

OUTPUT_TYPE = 'reg'
LBFGS = False

LEARNING_RATE = 1. / len(train_iter)
alpha_0 = 0.5 * len(train_iter)
EMBEDDING_SIZE = 12
N_VARIATIONAL_SAMPLES = 1

best_lr, best_a0, best_es = float('nan'), float('nan'), float('nan')
best_rmse = float('inf')

search_size = 5

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn('[progress.description]'
             + '{task.fields[n_elbo]}: {task.fields[elbo]:.4f} '
             + '{task.fields[n_rmse]}: {task.fields[test_rmse]:.4f} '
             + '{task.fields[n_all]}: {task.fields[all_rmse]:.4f} '),
) as progress:
    batch_task = progress.add_task(
        'Running batches...',
        total=len(train_iter),
        **default_progress
    )
    training = progress.add_task(
        'VFM training...',
        total=N_EPOCHS,
        **default_progress
    )

    print(run(lr=LEARNING_RATE, alpha_0=alpha_0, embedding_size=EMBEDDING_SIZE,
              n_groups=N_GROUPS, group_sizes=[N, M],
              n_var_samples=N_VARIATIONAL_SAMPLES))

    # task = progress.add_task(description='Grid Search', total=search_size**2, **default_progress)
    # progress.update(task, n_elbo='lr', n_rmse='a0')

    # ress = []
    # XX = []
    # for es in range(1, 21):
    #     # ress.append([])
    #     # XX.append([])
    #     # for a0 in 10 ** np.linspace(-0.3, 0.3, search_size):
    #     progress.update(task, elbo=best_es, all_rmse=best_rmse)
    #     loss = run(lr=LEARNING_RATE, alpha_0=alpha_0,
    #         embedding_size=es, n_groups=N_GROUPS, group_sizes=[N, M],
    #         n_var_samples=N_VARIATIONAL_SAMPLES)
    #     if loss < best_rmse:
    #         best_rmse = loss
    #         # best_lr = lr
    #         # best_a0 = a0
    #         best_es = es
    #     ress.append(loss)
    #     XX.append(es)
    #     progress.update(task, advance=1)
    #     progress.reset(training)
    # print(ress)
    # print(XX)

# print(best_rmse, best_es, EMBEDDING_SIZE, N_VARIATIONAL_SAMPLES)





