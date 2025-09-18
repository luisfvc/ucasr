import argparse
import os
import random
import subprocess
import time

import numpy as np
import torch
import wandb

from ucasr.refine_cca import refine_cca
from ucasr.wrappers import create_train_loaders, get_model
from ucasr.utils.colored_printing import BColors
from ucasr.utils.metrics_tracker import initialize_metrics_summary, update_metrics_summary, print_epoch_stats
from ucasr.utils.train_utils import iterate_dataset
from ucasr.utils.utils import load_yaml, make_dir, set_remote_paths, load_pretrained_model


def train_model(args, wandb_sweep=False, sweep_run_name=None):

    args = set_remote_paths(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ensuring reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # setting model paths and tag
    make_dir(args.exp_root)
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
        elif args.ft_path:
            exp_path = args.ft_path
        make_dir(os.path.join(exp_path, train_tag))

        # model tag
        model_tag = 'msmd_att' if args.use_att else 'msmd'
        model_tag_path = os.path.join(exp_path, train_tag, model_tag)
        make_dir(model_tag_path)

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
        exp_path = args.exp_root
        make_dir(os.path.join(exp_path, train_tag))

        run_dir = os.path.join(exp_path, train_tag, sweep_run_name)
        make_dir(run_dir)

        exp_tag = f"_{args.audio_context}"
        exp_tag = f"{exp_tag}{'_audio' if args.finetune_audio else ''}{'_score' if args.finetune_score else ''}"
        exp_tag = f"{exp_tag}{'_mixed' if args.finetune_mixed else ''}"
        dump_path = os.path.join(run_dir, f'params{exp_tag}.pt')
        if args.ft_last_run:
            dump_path = os.path.join(run_dir, f'params{exp_tag}_lm.pt')

    # getting dataloaders
    tr_loader, va_loader, tr_eval_loader = create_train_loaders(args)

    # get model
    model, loss = get_model(args)

    # load pretrained models if finetuning
    model = load_pretrained_model(args, model, audio=True) if (args.finetune_audio or args.finetune_mixed) else model
    model = load_pretrained_model(args, model, audio=False) if args.finetune_score else model
    print(f'Saving model in {dump_path}')
    model.to(device)

    lr = args.lr
    if wandb_sweep:
        lr = float('{0:.0e}'.format(args.lr))
        print(f'Initial learning rate: {lr}')
    # get optimizer and scheduler
    # todo: create a wrapper function
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()

    # tracking training
    metrics_summary = initialize_metrics_summary()
    current_epoch_time = time.monotonic()
    last_improvement = 0
    refinement_rounds = args.refinement_rounds
    patience = args.init_patience

    for epoch in range(args.n_epochs):

        tr_loss = iterate_dataset(model, tr_loader, loss_function=loss, optimizer=optim, device=device)

        tr_metrics = iterate_dataset(model, tr_eval_loader, loss_function=loss, optimizer=None, device=device)
        tr_metrics['loss'] = tr_loss

        va_metrics = iterate_dataset(model, va_loader, loss_function=loss, optimizer=None, device=device)

        metrics_summary = update_metrics_summary(metrics_summary, epoch, tr_metrics, va_metrics)

        # check for improvement
        try:
            improvement = va_metrics['map'] >= max(metrics_summary['va_map'][:-1])
        except ValueError:
            improvement = True

        if improvement:
            last_improvement = 0
            best_epoch = epoch
            best_model = model.state_dict()
            best_optim = optim.state_dict()
            best_va_metrics = va_metrics

            if not args.no_dump_model:
                best_checkpoint = {'epoch': epoch, 'model_params': best_model, 'optim_state': best_optim}
                torch.save(best_checkpoint, dump_path)
        last_improvement += 1
        improvement_countdown = patience - last_improvement + 1

        print_epoch_stats(metrics_summary, current_epoch_time, patience=improvement_countdown)
        current_epoch_time = time.monotonic()

        if args.use_wandb or wandb_sweep:
            wandb_metrics = {metric: value[-1] for metric, value in metrics_summary.items()}
            wandb.log(wandb_metrics)

        if last_improvement > patience:
            print(BColors().colored("\nEarly Stopping!", BColors.WARNING))
            best = (best_epoch, best_va_metrics['loss'], best_va_metrics['mean_dist'], best_va_metrics['map'] * 100)
            status = "Best Epoch: %d, Validation Loss: %.5f: Dist: %.5f Map: %.2f" % best
            print(BColors().colored(status, BColors.WARNING))

            if refinement_rounds <= 0:
                # stop training when all the refinement steps are done
                break
            else:
                text = f'Loading best parameters and refining ({refinement_rounds}) with decreased learning rate...'
                print(BColors().colored(text, BColors.WARNING))

                # decrease refinement steps
                patience = args.ref_patience
                last_improvement = 0
                refinement_rounds -= 1

                # load the best model so far
                model.load_state_dict(best_model)
                optim.load_state_dict(best_optim)

                # decrease learning rate
                lr *= args.lr_factor
                for g in optim.param_groups:
                    g['lr'] = lr

    if args.refine_cca:
        refine_cca(args, wandb_sweep=wandb_sweep, sweep_run_name=sweep_run_name)

    if args.eval_all:

        sh_args = ['python', 'eval_snippet_retrieval.py', '--ret_dir', 'both', '--refine_cca', '--dump_results']
        sh_args.append('--finetune_audio') if args.finetune_audio else None
        sh_args.append('--finetune_score') if args.finetune_score else None
        sh_args.append('--finetune_mixed') if args.finetune_mixed else None
        sh_args.append('--ft_last_run') if args.ft_last_run else None
        sh_args.extend(['--exp_path', exp_path])
        for dataset in ['MSMD', 'RealScores_Synth', 'RealScores_Rec']:
            ret = subprocess.call(sh_args + ['--dataset', dataset], shell=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')
    parser.add_argument('--ds_ratio', help='use a subset of the msmd', type=str, default=None)
    parser.add_argument('--use_att', help='use attention layer', action='store_true', default=False)
    parser.add_argument('--finetune_audio', help='load pretrained audio encoder', action='store_true', default=False)
    parser.add_argument('--finetune_score', help='load pretrained score encoder', action='store_true', default=False)
    parser.add_argument('--finetune_mixed', help='load pretrained mixed encoder', action='store_true', default=False)
    parser.add_argument('--no_dump_model', help='save best model every epoch', action='store_true', default=False)
    parser.add_argument('--eval_all', help='evaluate snippet retrieval after train', action='store_true', default=False)
    parser.add_argument('--refine_cca', help='refine cca layer after training', action='store_true', default=False)
    parser.add_argument('--ft_audio_path', help='location of the finetuned audio model', type=str, default=None)
    parser.add_argument('--ft_score_path', help='location of the finetuned score model', type=str, default=None)
    parser.add_argument('--ft_mixed_path', help='location of the finetuned mixed model', type=str, default=None)
    parser.add_argument('--ft_path', help='location to save the finetuned models', type=str, default=None)
    parser.add_argument('--ft_last_run', help='finetune last model', action='store_true', default=False)
    parser.add_argument('--use_wandb', help='monitor training with wandb', action='store_true', default=False)
    parser.add_argument('--run_name', help='wandb run name', type=str, default='')
    parser.add_argument('--separate_run', help='save run separately according to run_name', action='store_true',
                        default=False)

    configs = load_yaml('config/msmd_config.yaml')

    if parser.parse_args().use_wandb:
        wandb.init(config=configs, project='ucasr')
        if parser.parse_args().run_name:
            wandb.run.name = parser.parse_args().run_name

    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    train_model(parser.parse_args())
