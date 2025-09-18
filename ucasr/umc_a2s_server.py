import argparse
import os
import random
import sys

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
import yaml

from ucasr.audio_sheet_server import AudioSheetServer
from ucasr.wrappers import get_model
from ucasr.snippet_dataset import load_msmd_dataset, load_msmd_piece
from ucasr.utils.utils import load_yaml, set_remote_paths
from ucasr.utils.colored_printing import BColors

col = BColors()


class UMCSnippetDataset(TorchDataset):

    def __init__(self, scores, spectrograms, pieces, args, a2s=True):

        self.scores = scores
        self.spectrograms = spectrograms
        self.pieces = pieces
        self.a2s = a2s

        self.spec_context = int(args.snippet_len[args.audio_context] * args.fps)
        self.spec_bins = args.spec_bins
        self.sheet_context = args.sheet_context
        self.staff_height = args.staff_height

        self.entities = self.compute_score_entities() if a2s else self.compute_audio_entities()

    def compute_score_entities(self):

        score_entities = []
        for pid, piece_score in enumerate(self.scores):
            indices = np.arange(0, piece_score.shape[1] - self.sheet_context, 50)
            score_entities.extend([{'pid': pid, 'start': s} for s in indices])

        return score_entities

    def compute_audio_entities(self):

        audio_entities = []
        for pid, piece_spec in enumerate(self.spectrograms):
            indices = np.arange(0, piece_spec.shape[1] - self.spec_context, 10)
            audio_entities.extend([{'pid': pid, 'start': s} for s in indices])

        return audio_entities

    def prepare_audio_snippet(self, pid, start):

        stop = start + self.spec_context

        stop = np.min([stop, self.spectrograms[pid].shape[1] - 1])
        start = stop - self.spec_context

        spec_snippet = self.spectrograms[pid][:, start:stop]

        return spec_snippet

    def prepare_score_snippet(self, pid, x0):

        x1 = int(np.min([x0 + self.sheet_context, self.scores[pid].shape[1] - 1]))
        x0 = int(x1 - self.sheet_context)

        y0 = self.scores[pid].shape[0] // 2 - self.staff_height // 2
        y1 = y0 + self.staff_height

        sheet_snippet = self.scores[pid][y0:y1, x0:x1]

        return sheet_snippet

    def __getitem__(self, item):
        pid, start = self.entities[item]['pid'], self.entities[item]['start']

        if self.a2s:
            sheet_snippet = self.prepare_score_snippet(pid, start)
            sheet_snippet = cv2.resize(sheet_snippet / 255, (sheet_snippet.shape[1] // 2, sheet_snippet.shape[0] // 2))
            sheet_snippet = np.expand_dims(sheet_snippet.astype(np.float32), axis=0)

            spec_snippet = np.zeros((1, self.spec_bins, self.spec_context), dtype=np.float32)
            return sheet_snippet, spec_snippet
        else:
            spec_snippet = self.prepare_audio_snippet(pid, start)
            sheet_snippet = np.zeros((1, self.staff_height // 2, self.sheet_context // 2), dtype=np.float32)
            return sheet_snippet, np.expand_dims(spec_snippet, axis=0)

    @property
    def id_to_piece_name(self):
        return {pid: p_name for pid, p_name in enumerate(self.pieces)}

    @property
    def snippet_ids(self):
        return np.array([self.entities[item]['pid'] for item in range(len(self))])

    def __len__(self):
        return len(self.entities)


def do_umc_piece_retrieval(args, a2s_direction=True):

    print("\n--- Evaluating piece identification ---\n")

    args = set_remote_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # get model
    model, loss = get_model(args)
    model_params = torch.load(model_path)['model_params']
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()

    # load umc dataset
    splits = load_yaml(os.path.join(args.split_root, 'db_scanned_recording_piece_retrieval.yaml'))
    test_pieces = splits['test']
    scores = []
    specs = []
    for piece in test_pieces:
        npz_content = np.load(os.path.join(args.umc_root, f'{piece}.npz'), allow_pickle=True)
        scores.append(npz_content['unrolled_score'])
        specs.append(npz_content['performances'].item()['spec'])

    test_dataset = UMCSnippetDataset(scores, specs, test_pieces, args, a2s=a2s_direction)

    a2s_server = AudioSheetServer(model, test_dataset, device, args)
    a2s_server.initialize_db(is_sheet_db=a2s_direction)

    # run full evaluation
    if args.full_eval:
        print(col.colored('\nRunning full evaluation:', col.UNDERLINE))
        ranks = []

        for piece in a2s_server.test_pieces:
            npz_content = np.load(os.path.join(args.umc_root, f'{piece}.npz'), allow_pickle=True)
            spectrogram = npz_content['performances'].item()['spec']

            query_representation = spectrogram if a2s_direction else npz_content['unrolled_score']

            ret_result, ret_votes = a2s_server.detect_piece(query_representation, top_k=len(a2s_server.test_pieces),
                                                            detect_score=a2s_direction, n_candidates=args.n_candidates,
                                                            verbose=False)

            rank = ret_result.index(piece) + 1 if piece in ret_result else len(ret_result)
            ratio = ret_votes[ret_result.index(piece)] if piece in ret_result else 0.0

            ranks.append(rank)
            color = col.OKBLUE if ranks[-1] == 1 else col.WARNING
            print(col.colored(f'rank: {ranks[-1]:02d} ({ratio:.2f}) ', color) + piece)

        # report results
        modal = 'scores' if a2s_direction else 'performances'
        ranks = np.asarray(ranks)
        n_queries = len(ranks)
        for r in range(1, n_queries + 1):
            n_correct = int(np.sum(ranks == r))
            if n_correct > 0:
                print(col.colored(f'{n_correct} of {n_queries} retrieved {modal} ranked at position {r}.', col.WARNING))

        # dump retrieval results to file
        if args.dump_results:
            rd = "A2S"
            res_file = model_path.replace('params', 'piece_retrieval').replace('.pt', f'_{args.dataset}_{rd}{lm}.yaml')
            maps = float(np.mean([1.0 / r for r in ranks]))
            ranks = [int(r) for r in ranks]
            with open(res_file, 'w') as fp:
                yaml.dump({'mrr': maps, 'ranks': ranks}, fp, default_flow_style=False)

    else:
        # single eval
        test_piece = '193_Wolfgang_Amadeus_Mozart_-_Sonata_in_F_(KV_280),_1st_movement_187_Original_Score_1613'
        npz_content = np.load(os.path.join(args.umc_root, f'{test_piece}.npz'), allow_pickle=True)
        spectrogram = npz_content['performances'].item()['spec']
        query_representation = spectrogram if a2s_direction else npz_content['unrolled_score']

        modal = 'Audio' if a2s_direction else 'Score'
        print(col.colored(f'\nQuery {modal}: {test_piece}', color=col.OKBLUE))

        a2s_server.detect_piece(query_representation, top_k=5, n_candidates=args.n_candidates,
                                detect_score=a2s_direction, verbose=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--use_att', help='use attention layer', action='store_true', default=False)
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')
    parser.add_argument('--n_candidates', help='number of candidates per snippet', type=int, default=25)
    parser.add_argument('--dataset', help="evaluation configuration: 'MSMD', 'RealScores_Synth', 'RealScores_Rec', 'MSMD_Rec'",
                        type=str, default='MSMD')
    parser.add_argument('--dump_results', help='save results to file', action='store_true', default=False)
    parser.add_argument('--full_eval', help='evaluate entire dataset', action='store_true', default=False)
    parser.add_argument('--ret_dir', help='retrieval direction: a2s, s2a', type=str, default='a2s')
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

    do_umc_piece_retrieval(parser.parse_args(), a2s_direction=True)
