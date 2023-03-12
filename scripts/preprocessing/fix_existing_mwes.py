from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from resolve.training.data import read_training_sentences
from resolve.training.mwe_preproc.common import compute_output_path
from resolve.training.mwe_preproc.existing import process
from jsonlines import open as open_jsonl


def main():
    parser = ArgumentParser()
    parser.add_argument('sentences', type=Path)

    args = parser.parse_args()

    sentences = list(read_training_sentences(args.sentences, None))
    stats = process(tqdm(sentences, desc='Processing sentences'))

    output_path = compute_output_path(args.sentences, 'fixed')

    with open_jsonl(output_path, 'w') as outfile:
        for training_sentence in tqdm(sentences, f'Writing to {output_path}'):
            outfile.write(training_sentence.to_json())

    print('Done')
    print(stats)


if __name__ == '__main__':
    main()

