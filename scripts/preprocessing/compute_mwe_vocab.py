from argparse import ArgumentParser
from pathlib import Path

from resolve.training.data import read_training_sentences


def main():
    parser = ArgumentParser()
    parser.add_argument('input_file', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    input_path = args.input_file
    print(input_path)
    assert input_path.exists() and not input_path.is_dir()

    lemma_set = set()

    for training_sentence in read_training_sentences(input_path, None):
        for word in training_sentence:
            sense_data = word.mwe_sense_data
            if sense_data is not None:
                lemma = sense_data.lemma.strip()
                lemma_set.add(lemma)

    print(f'Found {len(lemma_set)} lemmas')

    with args.output.open('w') as f:
        f.writelines(f'{lem}\n' for lem in lemma_set)

    print(f'Done writing to {args.output}')


if __name__ == '__main__':
    main()
