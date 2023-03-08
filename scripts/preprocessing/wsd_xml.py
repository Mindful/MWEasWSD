from argparse import ArgumentParser
from pathlib import Path

from resolve.training.wsd import preprocess_xml


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=Path)

    args = parser.parse_args()
    preprocess_xml(args.input)


if __name__ == '__main__':
    main()

