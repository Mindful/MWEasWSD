from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from pathlib import Path

from tqdm import tqdm
from jsonlines import open as json_open

from resolve.training.data import fast_linecount
from resolve.training.mwe import DimSumSentence, DimSumWord


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    filepath: Path = args.input
    assert filepath.exists(), 'Input path must exist'

    lines = fast_linecount(str(filepath.absolute()))
    file_iter = filepath.open('r')

    DimsumLine = namedtuple('DimsumLine', ['idx', 'word', 'lemma', 'pos', 'mwe_tag', 'parent_idx',
                                           'unused', 'supersense', 'sent_id'])

    sentence_groupings = defaultdict(list)

    for line in tqdm(file_iter, total=lines):
        line = line.strip().split('\t')
        if len(line) > 1:
            line_data = DimsumLine(*line)
            sentence_groupings[line_data.sent_id].append(line_data)

    sentences = [
        DimSumSentence([DimSumWord(*line, manager=None) for line in lines], manager=None)
        for lines in sentence_groupings.values()
    ]

    with json_open(str(args.output.absolute()), 'w') as outfile:
        outfile.write_all(tqdm((s.to_json() for s in sentences), total=len(sentences),
                               desc=f'Writing outout to {args.output}'))


if __name__ == '__main__':
    main()
