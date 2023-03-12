from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from resolve.training.data import fast_linecount
from resolve.training.mwe import ConnlUWord, ConnlUSentence
from jsonlines import open as json_open

CUPT = 'cupt'
STREUSLE = 'streusle'


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--data_type', required=True, choices=[CUPT, STREUSLE])

    args = parser.parse_args()

    filepath: Path = args.input
    assert filepath.exists(), 'Input path must exist'
    sentences = []

    lines = fast_linecount(str(filepath.absolute()))
    line_buffer = []
    file_iter = filepath.open('r')

    for line in tqdm(file_iter, total=lines):
        line = line.strip()
        if line == '' and len(line_buffer) > 0:

            id_line = next(line for line in line_buffer
                           if line.startswith('# source_sent_id') or line.startswith('# sent_id'))
            original_text_line = next(line for line in line_buffer if line.startswith('# text '))
            word_lines = [l for l in line_buffer if l[0] != '#']

            assert original_text_line[0] == id_line[0] == '#'
            original_text = original_text_line.split('text =')[-1].strip()
            sent_id = id_line.split()[-1]

            word_data = [word_line.split('\t') for word_line in word_lines]

            words = [
                ConnlUWord(*data, manager=None) for data in word_data
                if '-' not in data[0]  # ignore STREUSLE entries spanning multiple tokens
            ]

            sentences.append(ConnlUSentence(words, sent_id, original_text, None))
            line_buffer = []
        else:
            line_buffer.append(line)

    file_iter.close()

    with json_open(str(args.output.absolute()), 'w') as outfile:
        outfile.write_all(tqdm((s.to_json() for s in sentences), total=len(sentences),
                               desc=f'Writing outout to {args.output}'))

    print('Done')


if __name__ == '__main__':
    main()
