import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from ucasr.models.cca_layer import CCARefine
from ucasr.wrappers import create_train_loaders, get_model
from ucasr.utils.utils import load_yaml, make_dir, set_remote_paths


def refine_cca(args, verbose=True, wandb_sweep=False, sweep_run_name=None):

    print("\n--- Refining CCA projections ---\n")
    print(f'Refining model with {args.n_refine} snippet pais from the training dataset')

    args = set_remote_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensuring reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    if not wandb_sweep:
        # training tag: trained, finetuned or pretrained
        do_finetune = args.finetune_audio or args.finetune_score or args.finetune_mixed
        train_tag = 'finetuned_models' if do_finetune else 'trained_models'
        train_tag = f"{train_tag}{f'/{args.run_name}' if args.separate_run else ''}"

        exp_path = args.exp_root
        if args.ft_audio_path:
            exp_path = args.ft_audio_path
        elif args.ft_score_path:
            exp_path = args.ft_score_path
        elif args.ft_mixed_path:
            exp_path = args.ft_mixed_path

        # model tag
        model_tag = 'msmd_att' if args.use_att else 'msmd'
        model_tag_path = os.path.join(exp_path, train_tag, model_tag)

        # experiment tag
        exp_tag = f"_{args.audio_context}"
        exp_tag = f"{exp_tag}{'_audio' if args.finetune_audio else ''}{'_score' if args.finetune_score else ''}"
        exp_tag = f"{exp_tag}{'_mixed' if args.finetune_mixed else ''}"
        ds_ratio = f'_{args.ds_ratio}' if args.ds_ratio else ''
        dump_path = os.path.join(model_tag_path, f'params{exp_tag}{ds_ratio}.pt')
        if args.ft_last_run:
            dump_path = os.path.join(model_tag_path, f'params{exp_tag}_lm.pt')

    else:
        train_tag = 'wandb_sweeps'
        make_dir(os.path.join(args.exp_root, train_tag))

        run_dir = os.path.join(args.exp_root, train_tag, sweep_run_name)
        make_dir(run_dir)

        exp_tag = f"_{args.audio_context}"
        exp_tag = f"{exp_tag}{'_audio' if args.finetune_audio else ''}{'_score' if args.finetune_score else ''}"
        exp_tag = f"{exp_tag}{'_mixed' if args.finetune_mixed else ''}"
        dump_path = os.path.join(run_dir, f'params{exp_tag}.pt')
        if args.ft_last_run:
            dump_path = os.path.join(run_dir, f'params{exp_tag}_lm.pt')

    # getting dataloaders
    tr_loader = create_train_loaders(args, only_refine=True)

    # get model
    model, _ = get_model(args)
    model_params = torch.load(dump_path)['model_params']
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()

    print("\nGetting pre-CCA latent embeddings...")

    pre_cca_sheet_embs = torch.tensor([], device=device)
    pre_cca_spec_embs = torch.tensor([], device=device)

    for batch_scores, batch_specs in tqdm(tr_loader, ncols=70, total=len(tr_loader), leave=False):
        batch_scores, batch_specs = batch_scores.to(device), batch_specs.to(device)

        # computing the pre-cca embeddings for the batch
        with torch.set_grad_enabled(False):
            batch_sheet_embs, batch_spec_embs = model(batch_scores, batch_specs, return_pre_cca=True)

        # stacking onto the final tensor
        pre_cca_sheet_embs = torch.cat((pre_cca_sheet_embs, batch_sheet_embs))
        pre_cca_spec_embs = torch.cat((pre_cca_spec_embs, batch_spec_embs))

    # creating a separate instance of the CCA layer to re-estimate the projection matrices
    cca_layer = CCARefine(in_dim=args.emb_dim)
    cca_layer.to(device)
    cca_layer.eval()

    # re-estimating the U and V projection matrices
    print("Fitting CCA model...")
    coeffs = cca_layer(pre_cca_sheet_embs, pre_cca_spec_embs)
    coeffs = coeffs.detach().cpu().numpy()

    if verbose:
        print("\nCorrelation-Coeffs:  ", np.around(coeffs, 3))
        print("Canonical-Correlation:", np.sum(coeffs) / args.emb_dim)

    # re-setting model CCA projection weights
    with torch.no_grad():
        model.cca_layer.mean1.data = cca_layer.mean1
        model.cca_layer.mean2.data = cca_layer.mean2
        model.cca_layer.U.data = cca_layer.U.data
        model.cca_layer.V.data = cca_layer.V.data

    if not args.no_dump_model:
        lm = '_lm' if args.ft_last_run else ''
        if not wandb_sweep:
            model_path_est_UV = os.path.join(exp_path, train_tag, f'{model_tag}_est_UV')
            make_dir(model_path_est_UV)
            refined_dump_path = os.path.join(model_path_est_UV, f'params{exp_tag}{lm}{ds_ratio}.pt')
        else:
            refine_dump_name = f'params{exp_tag}{lm}_est_UV.pt'
            refined_dump_path = os.path.join(run_dir, refine_dump_name)
        torch.save({'model_params': model.state_dict()}, refined_dump_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--use_att', help='use attention layer', action='store_true', default=False)
    parser.add_argument('--ds_ratio', help='use a subset of the msmd', type=str, default=None)
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')
    parser.add_argument('--no_dump_model', help='save best model every epoch', type=bool, default=False)
    parser.add_argument('--finetune_audio', help='load pretrained audio encoder', action='store_true', default=False)
    parser.add_argument('--finetune_score', help='load pretrained score encoder', action='store_true', default=False)
    parser.add_argument('--finetune_mixed', help='load pretrained mixed encoder', action='store_true', default=False)
    parser.add_argument('--ft_audio_path', help='location of the finetuned audio model', type=str, default=None)
    parser.add_argument('--ft_score_path', help='location of the finetuned score model', type=str, default=None)
    parser.add_argument('--ft_mixed_path', help='location of the finetuned mixed model', type=str, default=None)
    parser.add_argument('--ft_last_run', help='refine last model', action='store_true', default=False)
    parser.add_argument('--run_name', help='wandb run name', type=str, default='')
    parser.add_argument('--separate_run', help='save run separately according to run_name', action='store_true',
                        default=False)

    configs = load_yaml('config/msmd_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    refine_cca(parser.parse_args())
