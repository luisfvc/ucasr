import argparse
import os
import random
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

from ucasr.wrappers import get_model
from ucasr.snippet_dataset import load_msmd_dataset, load_msmd_piece
from ucasr.utils.utils import load_yaml, set_remote_paths
from ucasr.utils.colored_printing import BColors

col = BColors()


class UMCSnippetDataset(Dataset):

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


class AudioSheetServer:

    def __init__(self, model, test_dataset, device, args):

        self.args = args
        self.model = model
        self.device = device

        self.spec_context = int(args.snippet_len[args.audio_context] * args.fps)
        self.spec_bins = args.spec_bins
        self.sheet_context = args.sheet_context
        self.staff_height = args.staff_height

        self.datasets = {'MSMD': 'msmd_split.yaml', 'RealScores_Synth': 'db_scanned_synth.yaml',
                         'RealScores_Rec': 'db_scanned_recording.yaml'}

        self.test_dataset = test_dataset
        self.test_pieces = test_dataset.pieces
        self.db_id_to_piece = test_dataset.id_to_piece_name
        self.db_snippet_ids = test_dataset.snippet_ids
        self.db_snippet_codes = None

    def detect_piece(self, unrolled_query, top_k=1, n_candidates=1, detect_score=True, verbose=False):

        n_samples = 100
        snippet_width = self.spec_context if detect_score else self.sheet_context

        start_indices = np.linspace(start=0, stop=unrolled_query.shape[1] - snippet_width, num=n_samples)
        start_indices = start_indices.astype(np.int)

        if detect_score:
            # collect spectrogram excerpts
            spec_snippets = [unrolled_query[:, idx:idx + self.spec_context] for idx in start_indices]

            spec_snippets_tensor = torch.tensor(spec_snippets, dtype=torch.float32, device=self.device).unsqueeze(1)

            # dummy sheet snippets
            sheet_snippets_tensor = torch.zeros((n_samples, 1, self.staff_height // 2, self.sheet_context // 2),
                                                dtype=torch.float32, device=self.device)

        else:
            # slice central part of unrolled sheet
            r0 = unrolled_query.shape[0] // 2 - self.staff_height // 2
            r1 = r0 + self.staff_height

            # collect spectrogram excerpts
            sheet_snippets = [unrolled_query[r0:r1, idx:idx + self.sheet_context] for idx in start_indices]
            sheet_snippets = [cv2.resize(i / 255, (i.shape[1] // 2, i.shape[0] // 2)) for i in sheet_snippets]

            sheet_snippets_tensor = torch.tensor(sheet_snippets, dtype=torch.float32, device=self.device).unsqueeze(1)

            # dummy score snippets
            spec_snippets_tensor = torch.zeros((n_samples, 1, self.spec_bins, self.spec_context), dtype=torch.float32,
                                               device=self.device)

        self.model.eval()
        codes = self.model(sheet_snippets_tensor, spec_snippets_tensor)[int(detect_score)]
        codes = codes.detach().cpu().numpy()

        piece_ids = self._retrieve_snippet_ids(codes, n_candidates)

        # count voting for each piece
        unique, counts = np.unique(piece_ids, return_counts=True)

        # return top k pieces
        sorted_count_idxs = np.argsort(counts)[::-1][:top_k]

        # report
        if verbose:
            print(col.colored("\nRetrieval Ranking:", color=col.UNDERLINE))
            for idx in sorted_count_idxs:
                print(f'pid: {unique[idx]:03d} ({counts[idx]:03d}): {self.db_id_to_piece[unique[idx]]}')
            print("")

        ret_result = [self.db_id_to_piece[unique[idx]] for idx in sorted_count_idxs]
        ret_votes = [counts[idx] for idx in sorted_count_idxs]
        ret_votes = np.asarray(ret_votes, dtype=float) / np.sum(ret_votes)

        return ret_result, ret_votes

    def _retrieve_snippet_ids(self, query_codes, n_candidates=1):
        """ retrieve k most similar sheet music snippets """

        # compute distance
        dists = cdist(self.db_snippet_codes, query_codes, metric="cosine").T

        # sort indices by distance
        sorted_idx = np.argsort(dists)[:, :n_candidates].flatten()

        # return piece ids
        return self.db_snippet_ids[sorted_idx]

    def compute_embeddings(self, return_score_embs=True):

        loader = DataLoader(dataset=self.test_dataset, batch_size=256, drop_last=False, shuffle=False,
                            num_workers=self.args.n_workers)
        self.model.eval()

        # empty tensors for storing all embeddings)
        score_embs = torch.tensor([], device=self.device)
        audio_embs = torch.tensor([], device=self.device)

        for batch_x, batch_y in tqdm(loader, ncols=70, total=len(loader), leave=False):

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            with torch.set_grad_enabled(False):
                batch_x_embs, batch_y_embs = self.model(batch_x, batch_y)

            # stacking the batch embeddings into a single tensor
            score_embs = torch.cat((score_embs, batch_x_embs.detach()))
            audio_embs = torch.cat((audio_embs, batch_y_embs.detach()))

        assert score_embs.shape[0] == self.db_snippet_ids.shape[0]
        assert audio_embs.shape[0] == self.db_snippet_ids.shape[0]

        if return_score_embs:
            return score_embs.detach().cpu().numpy()
        return audio_embs.detach().cpu().numpy()

    def initialize_db(self, is_sheet_db=True):

        modal = 'sheet' if is_sheet_db else 'audio'

        print(f'\nInitializing {self.args.dataset} {modal} db...')

        self.db_snippet_codes = self.compute_embeddings(return_score_embs=is_sheet_db)
        print(f'{self.db_snippet_codes.shape[0]} {modal} codes of {len(self.db_id_to_piece)} pieces collected\n')


def do_piece_retrieval(args):

    print("\n--- Evaluating piece identification ---\n")

    # ensuring reproducibility
    seed = 456
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args = set_remote_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a2s_direction = args.ret_dir == 'A2S'

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
    msmd_ratio = f'_{args.msmd_ratio}' if args.msmd_ratio else ''
    exp_tag = f"{exp_tag}{lm}{msmd_ratio}"
    model_path = os.path.join(model_path, f'params{exp_tag}.pt')

    # get model
    model, loss = get_model(args)
    model_params = torch.load(model_path)['model_params']
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()
    print(f'Model Path: {model_path}')

    rd = 'A2S' if a2s_direction else 'S2A'
    print(f'\nRunning {rd} piece retrieval')

    # load test dataset for piece retrieval
    datasets = {'MSMD': 'msmd_split.yaml', 'RealScores_Synth': 'db_scanned_synth_piece_retrieval.yaml',
                'RealScores_Rec': 'db_scanned_recording_piece_retrieval.yaml',
                'RealScores_Synth_100': 'db_scanned_synth_piece_retrieval_100.yaml',
                'RealScores_Rec_100': 'db_scanned_recording_piece_retrieval_100.yaml'}

    if args.dataset == 'MSMD':
        splits = load_yaml(os.path.join(args.split_root, datasets[args.dataset]))
        test_split = splits['test'] + splits['test_v1']
        test_dataset = load_msmd_dataset(args.msmd_root, test_split, args, aug='test_aug')
    else:
        splits = load_yaml(os.path.join(args.split_root, datasets[args.dataset]))
        test_pieces = splits['test']
        scores = []
        specs = []
        for piece in test_pieces:
            npz_content = np.load(os.path.join(args.umc_root, f'{piece}.npz'), allow_pickle=True)
            scores.append(npz_content['unrolled_score'])
            specs.append(npz_content['performances'].item()['spec'])

        test_dataset = UMCSnippetDataset(scores, specs, test_pieces, args, a2s=a2s_direction)

    # initialize server
    a2s_server = AudioSheetServer(model, test_dataset, device, args)
    a2s_server.initialize_db(is_sheet_db=a2s_direction)

    # run full evaluation
    if args.full_eval:
        print(col.colored('\nRunning full evaluation:', col.UNDERLINE))
        ranks = []
        matching_qualities = []
        for piece in a2s_server.test_pieces:

            query_representation = get_query_representation(piece, a2s_direction, args)

            ret_result, ret_votes = a2s_server.detect_piece(query_representation, top_k=len(a2s_server.test_pieces),
                                                            detect_score=a2s_direction, n_candidates=args.n_candidates,
                                                            verbose=False)

            rank = ret_result.index(piece) + 1 if piece in ret_result else len(ret_result)
            ratio = ret_votes[ret_result.index(piece)] if piece in ret_result else 0.0001

            mqual = ret_votes[1] / ret_votes[0] if rank == 1 else ret_votes[0] / ratio
            matching_qualities.append(mqual)
            ranks.append(rank)
            color = col.OKBLUE if ranks[-1] == 1 else col.WARNING
            print(col.colored(f'rank: {ranks[-1]:02d} ({mqual:.2f}) ', color) + piece)

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
            res_file = model_path.replace('params', 'piece_retrieval').replace('.pt', f'_{args.dataset}_{rd}{lm}.yaml')
            # mqual_file = model_path.replace('params', 'mqual').replace('.pt', f'_{args.dataset}_{rd}{lm}.npz')
            # np.savez(mqual_file, mqual=matching_qualities)
            maps = float(np.mean([1.0 / r for r in ranks]))
            ranks = [int(r) for r in ranks]
            with open(res_file, 'w') as fp:
                yaml.dump({'mrr': maps, 'ranks': ranks}, fp, default_flow_style=False)

    else:
        # single eval
        test_piece = 'BachJS__BWVAnh691__BWV-691'
        query_representation = get_query_representation(test_piece, a2s_direction, args)

        modal = 'Audio' if a2s_direction else 'Score'
        print(col.colored(f'\nQuery {modal}: {test_piece}', color=col.OKBLUE))

        a2s_server.detect_piece(query_representation, top_k=5, n_candidates=args.n_candidates,
                                detect_score=a2s_direction, verbose=True)


def get_query_representation(piece_name, a2s, args):

    if args.dataset == 'MSMD':
        unrolled_score, perfs = load_msmd_piece(args.msmd_root, piece_name, args, aug='test_aug')
        spectrogram = perfs[0]['spec']

        query_representation = spectrogram if a2s else unrolled_score
        return query_representation

    npz_content = np.load(os.path.join(args.umc_root, f'{piece_name}.npz'), allow_pickle=True)
    spectrogram = npz_content['performances'].item()['spec']

    query_representation = spectrogram if a2s else npz_content['unrolled_score']
    return query_representation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_context', help='audio snippet size', type=str, default='mc')
    parser.add_argument('--use_att', help='use attention layer', action='store_true', default=False)
    parser.add_argument('--aug_config', help='augmentation configuration', type=str, default='full_aug')
    parser.add_argument('--n_candidates', help='number of candidates per snippet', type=int, default=25)
    parser.add_argument('--msmd_ratio', help='percentage of MSMD used for training', type=int, default=None)
    parser.add_argument('--dataset', help="evaluation configuration: 'MSMD', 'RealScores_Synth', 'RealScores_Rec', 'MSMD_Rec'",
                        type=str, default='MSMD')
    parser.add_argument('--dump_results', help='save results to file', action='store_true', default=False)
    parser.add_argument('--full_eval', help='evaluate entire dataset', action='store_true', default=False)
    parser.add_argument('--ret_dir', help='retrieval direction: a2s, s2a', type=str, default='A2S')
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

    do_piece_retrieval(parser.parse_args())
