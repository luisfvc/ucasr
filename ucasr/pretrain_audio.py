import argparse
import os
import random
import time

import numpy as np
import torch
import wandb

from ucasr.wrappers import create_maestro_loaders, get_model
from ucasr.utils.metrics_tracker import initialize_metrics_summary, update_metrics_summary, print_epoch_stats
from ucasr.utils.train_utils import iterate_dataset
from ucasr.utils.utils import load_yaml, make_dir, set_remote_paths


def pretrain_audio_model(args, wandb_sweep=False, sweep_run_name=None):

    args = set_remote_paths(args, audio_pretrain=True)

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
        # training tag: pretrained
        train_tag = f"pretrained_models{f'/{args.run_name}' if args.separate_run else ''}"

        run_dir = os.path.join(args.exp_root, train_tag)
        make_dir(run_dir)

        # experiment tag
        exp_tag = f'audio_{args.audio_context}'
        dump_path = os.path.join(args.exp_root, train_tag, f'params_{exp_tag}.pt')
        dump_path_lm = os.path.join(args.exp_root, train_tag, f'params_{exp_tag}_lm.pt')
    else:
        train_tag = 'wandb_sweeps'
        make_dir(os.path.join(args.exp_root, train_tag))

        run_dir = os.path.join(args.exp_root, train_tag, sweep_run_name)
        make_dir(run_dir)

        exp_tag = f'audio_{args.audio_context}'
        dump_path = os.path.join(run_dir, f'params_{exp_tag}.pt')
        dump_path_lm = os.path.join(run_dir, f'params_{exp_tag}_lm.pt')

    # getting dataloaders
    tr_loader, va_loader, tr_eval_loader = create_maestro_loaders(args)

    # get model
    model, loss = get_model(args, mode='audio_pretrain')
    model.to(device)

    lr = args.lr
    if wandb_sweep:
        lr = float('{0:.0e}'.format(args.lr))
        print(f'Initial learning rate: {lr}')
    # get optimizer and scheduler
    # todo: create a wrapper function
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='max', factor=args.lr_factor,
                                                           patience=args.patience, min_lr=1e-6, verbose=True)

    # tracking training
    metrics_summary = initialize_metrics_summary()
    current_epoch_time = time.monotonic()

    for epoch in range(args.n_epochs):

        tr_loss = iterate_dataset(model, tr_loader, loss_function=loss, optimizer=optim, device=device,
                                  is_pretrain=True)

        tr_metrics = iterate_dataset(model, tr_eval_loader, loss_function=loss, optimizer=None, device=device,
                                     is_pretrain=True)
        tr_metrics['loss'] = tr_loss

        va_metrics = iterate_dataset(model, va_loader, loss_function=loss, optimizer=None, device=device,
                                     is_pretrain=True)

        metrics_summary = update_metrics_summary(metrics_summary, epoch, tr_metrics, va_metrics)

        # check for improvement
        try:
            improvement = va_metrics['map'] >= max(metrics_summary['va_map'][:-1])
        except ValueError:
            improvement = True

        if improvement and not args.no_dump_model:
            best_checkpoint = {'epoch': epoch, 'model_params': model.state_dict(), 'optim_state': optim.state_dict()}
            torch.save(best_checkpoint, dump_path)

        # saving last model
        last_checkpoint = {'epoch': epoch, 'model_params': model.state_dict(), 'optim_state': optim.state_dict()}
        torch.save(last_checkpoint, dump_path_lm)

        print_epoch_stats(metrics_summary, current_epoch_time, patience=0)
        current_epoch_time = time.monotonic()

        scheduler.step(va_metrics['map'])

        if args.use_wandb or wandb_sweep:
            wandb_metrics = {metric: value[-1] for metric, value in metrics_summary.items()}
            wandb.log(wandb_metrics)

    print('\nPretraining audio encoder finished!\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--no_dump_model', help='save best model every epoch', action='store_true', default=False)
    parser.add_argument('--use_wandb', help='monitor training with wandb', action='store_true', default=False)
    parser.add_argument('--run_name', help='wandb run name', type=str, default='default_run')
    parser.add_argument('--rir_mode', help='how to apply the rir augmentation', type=str, default='no_rir')
    parser.add_argument('--separate_run', help='save run separately according to run_name', action='store_true',
                        default=False)

    configs = load_yaml('config/maestro_config.yaml')

    if parser.parse_args().use_wandb:
        wandb.init(config=configs, project='ucasr')
        if parser.parse_args().run_name:
            wandb.run.name = parser.parse_args().run_name

    for k, v in configs.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    pretrain_audio_model(parser.parse_args())
