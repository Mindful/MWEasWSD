from argparse import ArgumentParser
from pathlib import Path

import torch

from resolve.model.pl_module import ContextDictionaryBiEncoder
from resolve.mwe.filter import AllowedVocabFilter
from resolve.training.mwe_eval import MWEEvalData


def main():
    parser = ArgumentParser()
    parser.add_argument('data', choices=list(x.value for x in MWEEvalData if x != 'kulkarni'))
    parser.add_argument('--model', type=Path, required=False, default=None)
    parser.add_argument('--nocuda', required=False, action='store_true', default=False)
    parser.add_argument('--mwe_vocab', type=Path, required=False)
    parser.add_argument('--ban_filters', nargs='+', type=str)
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--examples', type=Path)

    args = parser.parse_args()

    if args.model is None:
        print('Warning: running with no model specified, so using rule-based pipelines only')
        model = None
    else:
        model = ContextDictionaryBiEncoder.load_from_checkpoint(str(args.model))
        if torch.cuda.is_available() and not args.nocuda:
            model.cuda()

    args = parser.parse_args()
    evaluator = MWEEvalData(args.data).get_evaluator(model, examples=args.examples)
    if args.mwe_vocab is not None:
        evaluator.pipeline.filters.insert(0, AllowedVocabFilter(args.mwe_vocab))

    if args.ban_filters:
        old_filters = str([x.__class__.__name__ for x in evaluator.pipeline.filters])
        evaluator.pipeline.filters = [
            f for f in evaluator.pipeline.filters
            if f.__class__.__name__ not in set(args.ban_filters)
        ]
        new_filters = str([x.__class__.__name__ for x in evaluator.pipeline.filters])
        print('filters:', old_filters, '-', set(args.ban_filters), '=', new_filters)

    if 'cupt' in args.data or 'coam' in args.data:
        filename = f'{args.data}_hyp.cupt'  # This is not used
        iterator = evaluator.cupt_lines()
    elif 'streusle' in args.data:
        raise RuntimeError("STREUSLE implementation not finished and doesn't currently work")
        filename = f'{args.data}_hyp.autoid.conllulex'
        iterator = evaluator.streusle_lines()
    elif 'dimsum' in args.data:
        filename = f'{args.data}_hyp.dimsum'
        iterator = evaluator.dimsum_lines()
    else:
        raise RuntimeError("Couldn't compute proper output filename")

    if args.output:
        filename = args.output

    with open(filename, 'w') as outfile:
        outfile.writelines(iterator)

    print('Wrote to', filename)


if __name__ == '__main__':
    main()
