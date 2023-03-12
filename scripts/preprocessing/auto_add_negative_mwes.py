import dataclasses
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from pprint import pp
from typing import Optional

import torch
from tqdm import tqdm

from resolve.model.pl_module import ContextDictionaryBiEncoder
from resolve.mwe.filter import OrderedOnly, InvertedFilter, MaxGappiness, UnionFilter
from resolve.mwe.index import MWEIndex
from resolve.mwe.pipeline import MWEPipeline
from resolve.mwe.resolve import NaiveResolver
from resolve.training.data import SenseData, NEGATIVE_MWE_GOLD_SENSE, read_training_sentences, compute_summary_stats, \
    to_percent
from resolve.training.mwe_eval import MWEEvalData
from resolve.training.mwe_preproc.new import all_new_candidates
from resolve.training.mwe_preproc.common import mwe_one_stdev_max, filter_annotations, AnnotationStatus, \
    apply_mwe_annotations, compute_output_path
from jsonlines import open as open_jsonl


class MaxCountCallback:
    def __init__(self, max_count: int):
        self.added_count = 0
        self.deleted_count = 0
        self.max_count = max_count

    def __call__(self, sense_data: SenseData) -> bool:
        if sense_data.metadata.get('annotated', False) and not sense_data.metadata.get('auto', False):
            # skip annotated data so we don't overwrite existing annotations as negatives
            return True
        elif self.added_count >= self.max_count:
            self.deleted_count += 1
            return False

        self.added_count += 1
        return True


def missing_for_percent(neg: int, pos: int, percent: float) -> int:
    return ceil(((neg * -percent) + neg - (pos * percent)) / (percent - 1))


def main():
    misordered_filter = InvertedFilter(OrderedOnly())
    overgapped_filter = InvertedFilter(MaxGappiness(5))
    false_types = {
        'misordered': misordered_filter,
        'overgapped': overgapped_filter,
        'all': UnionFilter(misordered_filter, overgapped_filter)
    }

    parser = ArgumentParser()
    parser.add_argument('input_path', type=Path)
    parser.add_argument('--false_type', choices=list(false_types.keys()), default='all')
    parser.add_argument('--target_percent', type=float, required=False)
    parser.add_argument('--pipeline', choices=list(x.value for x in MWEEvalData), required=False)
    parser.add_argument('--no_filter_candidates', action='store_true', default=False)
    parser.add_argument('--model', type=Path, required=False, default=None)

    args = parser.parse_args()

    if args.pipeline:
        if args.model:
            model = ContextDictionaryBiEncoder.load_from_checkpoint(str(args.model))
            if torch.cuda.is_available():
                model.cuda()
        else:
            model = None

        pipeline = MWEEvalData(args.pipeline).get_evaluator(model, None).pipeline
        print(f'Using {args.pipeline} pipeline, ignoring false_type argument')
    else:
        assert not args.model, 'Cannot use model with default pipeline'
        filters = [false_types[args.false_type]]
        index = MWEIndex.get_or_build_index(Path('temp_index.db'), in_memory=True)
        pipeline = MWEPipeline(index, filters, NaiveResolver())

    sentences = list(read_training_sentences(args.input_path, None))
    summary_stats = compute_summary_stats(sentences)
    print('Initial summary stats')
    pp(summary_stats.to_dict())

    stats, annotation_candidates = all_new_candidates(pipeline, tqdm(sentences, 'computing candidates'))
    for candidate in annotation_candidates:
        candidate.sense_data.gold_sense = NEGATIVE_MWE_GOLD_SENSE
        candidate.sense_data.metadata['annotated'] = True
        candidate.sense_data.metadata['auto'] = True

    annotation_candidates = [
        dataclasses.replace(c, status=AnnotationStatus.DONE_AUTO) for c in annotation_candidates
    ]

    print('Stats for additions (before filtering)')
    pp(stats)

    if not args.no_filter_candidates:
        max_mwe_count = mwe_one_stdev_max(sentences)
        len_before_filter = len(annotation_candidates)
        if args.target_percent:
            print(
                f'Limiting added negative MWE count to number required to reach {to_percent(args.target_percent)} negative')
            current_percent = summary_stats.mwe_active_negative / summary_stats.mwe_active

            if current_percent >= args.target_percent:
                print(f'Already at {to_percent(current_percent)} negatives,'
                      f' which is >= target of {to_percent(args.target_percent)}')
                return
            else:
                max_count = missing_for_percent(summary_stats.mwe_active_negative,
                                                summary_stats.mwe_active - summary_stats.mwe_active_negative,
                                                args.target_percent)

                print(f'Adding at most {max_count} more negatives')

            additional_filter = MaxCountCallback(max_count)
        else:
            additional_filter = None
        annotation_candidates = filter_annotations(annotation_candidates, max_mwe_count,
                                                   additional_filter=additional_filter)
        len_after_filter = len(annotation_candidates)
        print(f'Filtered additions from {len_before_filter} to {len_after_filter} - '
              f'max for any given MWE was {max_mwe_count}, '
              f'max total was {additional_filter.max_count if additional_filter is not None else "None"}')

    application_stats = apply_mwe_annotations(sentences, annotation_candidates)
    print('Applied annotations')
    print(application_stats)

    output_path = compute_output_path(args.input_path, 'autoneg')
    print(f'Writing annotations to {output_path}')
    with open_jsonl(output_path, 'w') as outfile:
        for training_sentence in sentences:
            outfile.write(training_sentence.to_json())

    print('Done. Final summary stats:')
    pp(compute_summary_stats(sentences).to_dict())


if __name__ == '__main__':
    main()
