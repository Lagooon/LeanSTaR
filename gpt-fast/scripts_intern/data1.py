from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from tqdm import tqdm
from pathlib import Path
import os
import json
import copy
import random

TOKEN_MAP = {
    'pad': '[PAD]',
    'eos': '<|endoftext|>',
}

PHRASE_MAP = {
    'goal': '[GOAL]',
    'proofstep': '[PROOFSTEP]',
}




def _load_ds(data_dir):
    ds = {}
    for split in ['train', 'val', 'test']:
        ds[split] = json.load(open(os.path.join(
            data_dir, 'leandojo_benchmark_4', 'novel_premises', f'{split}.json'), 'r')
        ) 
    return ds


def _save_splits(splits, data_dir, tag):
    print("Saving split to disk...")
    out_dir = os.path.join(data_dir, 'processed')
    for split, examples in tqdm(splits.items(), total=len(splits)):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(
            out_dir, '%s-%s.json' % (tag, split)
        )
        with open(out_file, 'w') as f:
            json.dump(examples, f, indent=4)


def _print_stats(splits):
    for split, examples in splits.items():
        print("%s\t%d" % (split, len(examples)))


def _fmt_proofstep(state_before, tactic):
    # [GOAL]{state_before}[PROOFSTEP]{tactic}<|endoftext|>
    inp = f"{PHRASE_MAP['goal']}{state_before}{PHRASE_MAP['proofstep']}"
    out = f"{tactic}{TOKEN_MAP['eos']}"
    return inp, out


def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts

def fmt_proofstep(split):
    examples = []
    a1 = 0
    dc = {}
    random.seed(42)
    random.shuffle(split)
    for traced_theorem in split:
        file_path = traced_theorem['file_path']
        full_name = traced_theorem['full_name']
        if file_path not in dc:
            dc[file_path] = 0
        dc[file_path] += 1
        if dc[file_path] > 5:
            continue
        random.shuffle(traced_theorem['traced_tactics'])
        
        for tactic_example in traced_theorem['traced_tactics'][:4]:
            examples.append({
                'state_before': tactic_example['state_before'],
                'state_after': tactic_example['state_after'],
                'tactic': tactic_example['tactic'],
                'file_path' : file_path,
                'full_name' : full_name,
            })
    return examples


def proofstep(data_dir):
    ds = _load_ds(data_dir)
    out_ds = {}
    for split in ds:
        out_ds[split] = fmt_proofstep(ds[split])
    
    _save_splits(
        splits=out_ds,
        data_dir=data_dir,
        tag='proofstep_g'
    )
    _print_stats(
        splits=out_ds
    )
    return out_ds


def main(args):
    proofstep(args.data_dir)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/leandojo_benchmark_4')

    args = parser.parse_args()
    main(args)