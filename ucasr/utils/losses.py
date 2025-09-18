import torch


def triplet_loss(x_batch, y_batch, margin=0.7):
    """ hinge loss function applied over entire batch using dot
    product as similarity function, because it assumes the
    embeddings are already normalized """

    # computing pairwise dot product similarity matrix
    pairwise_sims = torch.mm(x_batch, y_batch.t())

    # distances to be minimized are in the diagonal of pairwise_cos_dists (matching samples)
    eye_matrix = torch.eye(x_batch.shape[0], dtype=torch.bool, device=pairwise_sims.device)
    dists_minimize = pairwise_sims[eye_matrix].unsqueeze(-1)

    # exclude diagonal entries
    dists = (dists_minimize - pairwise_sims)[~eye_matrix]

    # computing the hinge loss
    loss = torch.clamp(margin - dists, 0, 1000)

    # n_positive_triplets = torch.count_nonzero(loss)
    # fraction_positive_triplets = n_positive_triplets / loss.shape[0]

    # alternative_loss = loss.sum() / (n_positive_triplets.float() + 1e-16)

    # return loss.mean(), fraction_positive_triplets
    return loss.mean()


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    credits: https://github.com/PyTorchLightning/lightning-bolts/blob/2415b49a2b405693cd499e09162c89f807abbdc4
    /pl_bolts/models/self_supervised/simclr/simclr_module.py#L223
    """

    out = torch.cat([z_i, z_j], dim=0)
    n_samples = len(out)

    # full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # positive similarity
    pos = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()
    return loss
