from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

from jsonlines import open as open_jsonl
from tqdm import tqdm

from resolve.training.data import fast_linecount, read_training_sentences, NEGATIVE_MWE_GOLD_SENSE
from resolve.training.mwe_preproc.common import MWEAnnotation, ANNOTATION_KEY, compute_output_path, \
    apply_mwe_annotations


@dataclass
class AnnotationStats:
    duplicates: int = 0
    accepted: int = 0
    not_accepted: int = 0
    pos_example: int = 0
    neg_example: int = 0

    def __str__(self):
        return str(asdict(self))


def get_prodigy_annotations_to_apply(filepath: str) -> Tuple[List[MWEAnnotation], AnnotationStats]:
    stats = AnnotationStats()
    annotation_id_set = set()
    output_annotations = []
    total = fast_linecount(filepath)

    for prodigy_entry in tqdm(open_jsonl(filepath), total=total, desc=filepath):
        if prodigy_entry['answer'] == 'accept':
            stats.accepted += 1
            annotation = MWEAnnotation.from_json(prodigy_entry[ANNOTATION_KEY])
            assert annotation.sense_data.gold_sense is not None
            annotation.sense_data.metadata['annotated'] = True
            if annotation.annotation_id in annotation_id_set:
                stats.duplicates += 1
            else:
                annotation_id_set.add(annotation.annotation_id)
                output_annotations.append(annotation)
                if annotation.sense_data.gold_sense == NEGATIVE_MWE_GOLD_SENSE:
                    stats.neg_example += 1
                else:
                    stats.pos_example += 1
        else:
            stats.not_accepted += 1

    return output_annotations, stats


def main():
    parser = ArgumentParser()
    parser.add_argument('annotations')
    parser.add_argument('sentences', type=Path)

    args = parser.parse_args()

    annotation_list, annotation_stats = get_prodigy_annotations_to_apply(args.annotations)
    print('Loaded annotations')
    print(annotation_stats)

    output_path = compute_output_path(args.sentences, 'annotated')

    training_sentences = list(read_training_sentences(args.sentences, None))
    application_stats = apply_mwe_annotations(training_sentences, annotation_list)
    print('Applied annotations')
    print(application_stats)

    print(f'Writing annotations to {output_path}')
    with open_jsonl(output_path, 'w') as outfile:
        for training_sentence in training_sentences:
            outfile.write(training_sentence.to_json())

    print('Done')


if __name__ == '__main__':
    main()
