import numpy as np
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm


def compute_metrics(x, y, expand_metrics=False):

    # getting number of samples for each view
    n_x = x.shape[0]

    if not expand_metrics:
        # computing pairwise distances
        dists = cdist(x, y, metric='cosine')

        sorted_indices = np.argsort(dists)
        boolean_rank_matrix = np.expand_dims(np.arange(n_x), axis=1) == sorted_indices
        ranks = (boolean_rank_matrix.nonzero()[1] + 1).tolist()
    else:
        z = np.vstack((x, y))
        n_z = z.shape[0]

        dists = cdist(z, z, metric='cosine')

        mask = ~np.eye(n_z, dtype=bool)
        masked_dists = dists[mask].reshape(n_z, -1)

        sorted_indices = np.argsort(masked_dists)
        diag_fixer = np.expand_dims(np.arange(n_z), axis=1)

        sorted_indices[sorted_indices >= diag_fixer] += 1
        sorted_indices = sorted_indices % n_x

        rank_vec = np.expand_dims(np.arange(n_x), axis=1)

        boolean_rank_matrix = np.vstack((rank_vec, rank_vec)) == sorted_indices
        ranks = (boolean_rank_matrix.nonzero()[1] + 1).tolist()

    # hit rates at 1, 5, 10 and 25
    hit_rates = {1: 0, 5: 0, 10: 0, 25: 0}

    for key in hit_rates.keys():
        # count how many items are in the top k
        hit_rates[key] = sum(1 for k in ranks if k <= key)

    # average precision
    aps = [1.0 / r for r in ranks]

    # compute some stats
    mean_rank = np.mean(ranks)  # mean rank
    median_rank = np.median(ranks)  # median rank
    mean_dist = np.diag(dists).mean()  # average sheet-spec distance in the latent space for matching pairs
    maps = np.mean(aps)  # mean average precision (mean reciprocal rank)

    # wrapping metrics into a dictionary
    stats = dict(
        mean_rank=mean_rank,
        med_rank=median_rank,
        mean_dist=mean_dist,
        map=maps,
        hit_rates=hit_rates,
        ranks=ranks,
        cos_dists=dists.diagonal() if not expand_metrics else 0
    )

    return stats


def iterate_dataset(model, dataloader, loss_function, optimizer=None, device=torch.device('cpu'),
                    retrieval_direction='a2s', is_pretrain=False, mixed_pretrain=False):
    """
    retrieval_direction: 'a2s' for audio-to-score query
                         's2a' for score-to-audio query
                         'both' for evaluating both search directions
    when multimodal training, x stands for sheet music snippets and y for audio spectrogram excerpts
    """

    is_train = optimizer is not None

    model.train() if is_train else model.eval()

    # empty tensors for storing all embeddings from this epoch (only in eval mode)
    x_embs = torch.tensor([], device=device)
    y_embs = torch.tensor([], device=device)

    losses = []

    for batch_x, batch_y in tqdm(dataloader, ncols=70, total=len(dataloader), leave=False):

        if mixed_pretrain:
            batch_x_0, batch_x_1 = batch_x[0].to(device), batch_x[1].to(device)
            batch_y_0, batch_y_1 = batch_y[0].to(device), batch_y[1].to(device)
            del batch_x
            del batch_y
            torch.cuda.empty_cache()
        else:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.set_grad_enabled(is_train):

            if not is_pretrain:
                batch_x_embs, batch_y_embs = model(batch_x, batch_y)
            elif mixed_pretrain:
                # batch_x_embs_0, batch_y_embs_0 = model(batch_x_0, batch_y_0)
                # batch_x_embs_1, batch_y_embs_1 = model(batch_x_1, batch_y_1)
                batch_x_embs = torch.cat(model(batch_x_0, batch_y_0))
                batch_y_embs = torch.cat(model(batch_x_1, batch_y_1))
            else:
                batch_x_embs = model(batch_x)
                batch_y_embs = model(batch_y)

            # compute loss
            loss = loss_function(batch_x_embs, batch_y_embs)

        # append current loss
        losses.append(loss.item())

        # compute the gradients and update the weights
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # stacking the batch embeddings into a single tensor for collecting results
        if not is_train:
            x_embs = torch.cat((x_embs, batch_x_embs.detach()))
            y_embs = torch.cat((y_embs, batch_y_embs.detach()))

    if is_train:
        return np.mean(losses)

    if retrieval_direction == 'both':
        a2s_metrics = compute_metrics(y_embs.cpu().numpy(), x_embs.cpu().numpy())
        s2a_metrics = compute_metrics(x_embs.cpu().numpy(), y_embs.cpu().numpy())
        a2s_metrics['loss'] = s2a_metrics['loss'] = np.mean(losses)
        return s2a_metrics, a2s_metrics

    if retrieval_direction == 'a2s':
        metrics = compute_metrics(y_embs.cpu().numpy(), x_embs.cpu().numpy(), expand_metrics=mixed_pretrain)
    elif retrieval_direction == 's2a':
        metrics = compute_metrics(x_embs.cpu().numpy(), y_embs.cpu().numpy())
    else:
        print("Please use 'a2s', 's2a' or 'both' for the retrieval direction")
        return None

    metrics['loss'] = np.mean(losses)
    return metrics
