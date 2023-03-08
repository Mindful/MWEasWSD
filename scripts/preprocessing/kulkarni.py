from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from resolve.training.data import fast_linecount
from resolve.training.mwe import KulkarniSentence
from jsonlines import open as json_open


END_LINE = '-' * 8


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=Path)

    args = parser.parse_args()

    filepath: Path = args.input
    assert filepath.exists(), f'Input path must exist; got input={filepath}'
    sentences = []

    lines = fast_linecount(str(filepath.absolute()))
    line_buffer = []
    for line in tqdm(filepath.open('r'), total=lines):
        line = line.strip()
        if line == END_LINE:
            sentence = line_buffer[0]
            mwes = line_buffer[1:]
            sentences.append(KulkarniSentence(sentence, mwes, None))
            line_buffer = []
        else:
            line_buffer.append(line)

    output_path = filepath.parent / 'kulkarni.jsonl'
    with json_open(str(output_path.absolute()), 'w') as outfile:
        outfile.write_all(tqdm((s.to_json() for s in sentences), total=len(sentences),
                               desc=f'Writing output to {output_path}'))

    print('Done')




if __name__ == '__main__':
    main()