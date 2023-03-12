from argparse import ArgumentParser
from pathlib import Path
from random import Random
from jsonlines import open as json_open
from tqdm import tqdm

from resolve.training.data import read_training_sentences, to_percent


def main():
    parser = ArgumentParser()
    parser.add_argument('file', type=Path)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--eval_percent', type=float, default=0.1)

    args = parser.parse_args()
    assert args.eval_percent < 1.0

    print('Random split', args.file, 'with seed', args.seed, 'and eval percent', to_percent(args.eval_percent))
    sentences = list(read_training_sentences(args.file, manager=None))

    rand = Random(args.seed)
    dev_count = int(len(sentences) * args.eval_percent)
    rand.shuffle(sentences)

    name_components = args.file.name.split('.')
    train_name = f'{name_components[0]}_train.{".".join(name_components[1:])}'
    train_output = args.file.parent / train_name
    dev_name = f'{name_components[0]}_dev.{".".join(name_components[1:])}'
    dev_output = args.file.parent / dev_name

    outputs = [
        (train_output, sentences[dev_count:]),
        (dev_output, sentences[:dev_count])
    ]
    for outpath, data in outputs:
        print(len(data), 'sentences for', outpath)
        with json_open(str(outpath.absolute()), 'w') as outfile:
            outfile.write_all(tqdm((s.to_json() for s in data), desc=f'Writing output to {outpath}'))


if __name__ == '__main__':
    main()
