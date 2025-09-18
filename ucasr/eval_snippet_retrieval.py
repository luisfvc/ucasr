import argparse
import os
import random

import numpy as np
import torch
import yaml

from ucasr.wrappers import create_test_loader, get_model
from ucasr.utils.train_utils import iterate_dataset
from ucasr.utils.utils import load_yaml, set_remote_paths


def eval_snippet_retrieval(args):

    print("\n--- Evaluating snippet retrieval ---\n")

    args = set_remote_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensuring reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # setting model paths and tag
    model_tag = f"msmd{'_att' if args.use_att else ''}{'_est_UV' if args.refine_cca else ''}"
    print(f'Model tag: {model_tag}')
    print(f'Audio context: {args.audio_context}')
    print(f'Evaluation data: {args.dataset}')

    do_finetune = args.finetune_audio or args.finetune_score or args.finetune_mixed
    train_tag = 'finetuned_models' if do_finetune else 'trained_models'
    train_tag = f"{train_tag}{f'/{args.run_name}' if args.separate_run else ''}"

    exp_path = args.exp_path if args.exp_path else args.exp_root

    model_path = os.path.join(exp_path, train_tag, model_tag)

    exp_tag = f"_{args.audio_context}"
    exp_tag = f"{exp_tag}{'_audio' if args.finetune_audio else ''}{'_score' if args.finetune_score else ''}"
    exp_tag = f"{exp_tag}{'_mixed' if args.finetune_mixed else ''}"
    lm = '_lm' if args.ft_last_run else ''
    model_path = os.path.join(model_path, f'params{exp_tag}{lm}.pt')

    # getting dataloaders
    te_loader = create_test_loader(args)
    print(f'Number of candidates from the test set: {args.n_test}')

    # get model
    model, loss = get_model(args)
    model_params = torch.load(model_path)['model_params']
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()

    te_metrics = iterate_dataset(model, te_loader, loss_function=loss, optimizer=None, device=device,
                                 retrieval_direction=args.ret_dir)

    if args.ret_dir == 'both':
        ret_dirs = ['s2a', 'a2s']
    else:
        te_metrics = [te_metrics]
        ret_dirs = [args.ret_dir]

    for metrics, rd in zip(te_metrics, ret_dirs):

        print("\nRetrieval direction: ", rd.upper())

        mean_rank_te = metrics['mean_rank']
        med_rank_te = metrics['med_rank']
        dist_te = metrics['mean_dist']
        hit_rates = metrics['hit_rates']
        mrr = metrics['map']
        ranks = metrics['ranks']
        cos_dists = metrics['cos_dists']

        # report hit rates
        recall_at_k = dict()

        print("\n\tHit Rates:")
        for key, hits in hit_rates.items():
            recall_at_k[key] = float(100 * hits) / args.n_test
            pk = recall_at_k[key] / key
            print(f'\tTop {key:02d}: {recall_at_k[key]:.3f} ({hits}) {pk:.3f}')

        print('\n')
        print(f'\tMedian Rank: {med_rank_te:.2f} ({args.n_test})')
        print(f'\tMean Rank  : {mean_rank_te:.2f} ({args.n_test})')
        print(f'\tMean Dist  : {dist_te:.5f}')
        print(f'\tMAP        : {mrr:.3f}')
        print('\n')

        if args.dump_results:

            mrr = float(mrr)
            med_rank_te = float(med_rank_te)
            keys = [f'{key}' for key in recall_at_k.keys()]
            recall_at_k = dict(zip(keys, recall_at_k.values()))

            results = {"map": mrr, 'med_rank': med_rank_te, 'recall_at_k': recall_at_k}

            rd = rd.upper()
            dump_file = model_path.replace('params', 'eval').replace('.pt', f'_{args.dataset}_{rd}{lm}.yaml')
            cos_dists_file = model_path.replace('params', 'cos_dists').replace('.pt', f'_{args.dataset}{lm}_{args.n_test}.npz')
            np.savez(cos_dists_file, cos_dists=cos_dists)

            # with open(dump_file, 'w') as f:
            #     yaml.dump(results, f, default_flow_style=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--use_att', help='use attention layer', action='store_true', default=False)
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')
    parser.add_argument('--dataset',
                        help="evaluation configuration: 'MSMD', 'RealScores_Synth', 'RealScores_Rec', 'MSMD_Rec'",
                        type=str, default='MSMD')
    parser.add_argument('--ret_dir', help='retrieval direction: a2s, s2a or both', type=str, default='a2s')
    parser.add_argument('--dump_results', help='save results to file', action='store_true', default=False)
    parser.add_argument('--refine_cca', help='evaluate for refined model', action='store_true', default=False)
    parser.add_argument('--finetune_audio', help='evaluate audio-finetuned model', action='store_true', default=False)
    parser.add_argument('--finetune_score', help='evaluate audio-finetuned model', action='store_true', default=False)
    parser.add_argument('--finetune_mixed', help='load pretrained mixed encoder', action='store_true', default=False)
    parser.add_argument('--ft_last_run', help='evaluate last model', action='store_true', default=False)
    parser.add_argument('--exp_path', help='optional path of the experiment', type=str, default=None)
    parser.add_argument('--separate_run', help='save run separately according to run_name', action='store_true',
                        default=False)
    parser.add_argument('--run_name', help='wandb run name', type=str, default='')


    configs = load_yaml('config/msmd_config.yaml')
    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    eval_snippet_retrieval(parser.parse_args())
