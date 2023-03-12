from argparse import ArgumentParser
from pathlib import Path
from pprint import pp
from typing import Optional

from jsonlines import open as json_open
from tqdm import tqdm

from resolve.mwe.filter import OrderedOnly, MaxGappiness, AllowedVocabFilter
from resolve.mwe.index import MWEIndex
from resolve.mwe.pipeline import MWEPipeline
from resolve.mwe.resolve import NaiveResolver
from resolve.training.data import read_training_sentences
from resolve.training.mwe_preproc.common import mwe_one_stdev_max, filter_annotations
from resolve.training.mwe_preproc.new import all_new_candidates


def get_pipeline(vocab_path: Optional[Path]) -> MWEPipeline:
    filters = [OrderedOnly(), MaxGappiness(3)]
    if vocab_path is not None:
        filters.insert(0, AllowedVocabFilter(vocab_path))

    index = MWEIndex.get_or_build_index(Path('temp_index.db'), in_memory=True)
    pipeline = MWEPipeline(index, filters, NaiveResolver())

    return pipeline


def main():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=Path)
    parser.add_argument('--mwe_vocab', type=Path, required=False)
    parser.add_argument('--add_mwes', default=False, action='store_true')

    args = parser.parse_args()

    pipeline = get_pipeline(args.mwe_vocab)

    sentences = list(read_training_sentences(args.input_path, None))
    stats, candidates = all_new_candidates(pipeline, tqdm(sentences, 'computing candidates'))
    print('Stats for candidates (before filtering)')
    pp(stats)
    max_mwe_count = mwe_one_stdev_max(sentences)
    len_before_filter = len(candidates)
    candidates = filter_annotations(candidates, max_mwe_count)
    len_after_filter = len(candidates)
    print(f'Filtered candidates from {len_before_filter} to {len_after_filter} - '
          f'max for any given MWE was {max_mwe_count}')

    output_path = args.input_path.parent / 'annotation_candidates.jsonl'
    with json_open(output_path, 'w') as out_file:
        out_file.write_all(cand.to_json() for cand in candidates)

    print(f'Wrote filtered candidates to {output_path}')


if __name__ == '__main__':
    main()
