from argparse import ArgumentParser
from pathlib import Path
from random import Random
from jsonlines import open as json_open
from tqdm import tqdm

from resolve.training.data import read_training_sentences


def main():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=False)

    args = parser.parse_args()
    print('Random sampling', args.count, 'from', args.file, 'with seed', args.seed)
    sentences = list(read_training_sentences(args.file, manager=None))

    rand = Random(args.seed)
    random_sample = rand.sample(sentences, args.count)

    if args.output:
        output = args.output
    else:
        name_components = args.file.name.split('.')
        new_name = f'{name_components[0]}_sample.{".".join(name_components[1:])}'
        output = args.file.parent / new_name

    with json_open(str(output.absolute()), 'w') as outfile:
        outfile.write_all(tqdm((s.to_json() for s in random_sample), desc=f'Writing output to {output}'))

    print('Done')


if __name__ == '__main__':
    main()
