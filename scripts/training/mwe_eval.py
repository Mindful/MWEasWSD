from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from pprint import pp

import torch
import wandb

from resolve.model.pl_module import ContextDictionaryBiEncoder
from resolve.mwe.filter import AllowedVocabFilter
from resolve.training.mwe_eval import MWEEvalData


def main():
    parser = ArgumentParser()
    parser.add_argument('data', choices=list(x.value for x in MWEEvalData))
    parser.add_argument('--max', required=False, type=int, default=None)
    parser.add_argument('--model', type=Path, required=False, default=None)
    parser.add_argument('--nocuda', required=False, action='store_true', default=False)
    parser.add_argument('--nowandb', required=False, action='store_true', default=False)
    parser.add_argument('--flags', required=False)  # no_filters, print
    parser.add_argument('--mwe_vocab', type=Path, required=False)

    args = parser.parse_args()

    if args.model is None:
        print('Warning: running with no model specified, so using rule-based pipelines only')
        model = None
    else:
        model = ContextDictionaryBiEncoder.load_from_checkpoint(str(args.model))
        if torch.cuda.is_available() and not args.nocuda:
            model.cuda()

    if args.flags:
        kwargs = {flag: True for flag in args.flags.split(',')}
    else:
        kwargs = {}

    evaluator = MWEEvalData(args.data).get_evaluator(model, args.max, **kwargs)
    if args.mwe_vocab is not None:
        evaluator.pipeline.filters.insert(0, AllowedVocabFilter(args.mwe_vocab))

    print('Index metadata')
    index_metadata = evaluator.pipeline.index.metadata()
    pp(index_metadata)
    print('Pipeline metadata')
    pipeline_metadata = evaluator.pipeline.metadata()
    pp(pipeline_metadata)

    if not args.nowandb:
        wandb.init(project="mwe_pipeline", entity="resolve", config={
            'index': index_metadata,
            'pipeline': pipeline_metadata,

        })
        wandb.config.update(args)

    aligned_results, binary_results, pipeline_results = evaluator()
    print('Aligned results:')
    print(aligned_results)
    print('-----------------')
    print('Binary results:')
    print(binary_results)
    print('-----------------')
    print('Pipeline results:')
    pp(pipeline_results)

    if not args.nowandb:
        for key, val in {
            **{
                f'aligned/{key}': val for key, val in asdict(aligned_results).items()
            },
            **{
                f'binary/{key}': val for key, val in asdict(binary_results).items()
            }
        }.items():
            wandb.run.summary[key] = val


if __name__ == '__main__':
    main()
