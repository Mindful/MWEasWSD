from __future__ import annotations

import random
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum
from math import ceil
from pathlib import Path
from statistics import mean, stdev
from typing import List, Optional, Callable, Tuple, Iterable, Dict
from jsonlines import open as json_open
from tqdm import tqdm

from resolve.common.data import SENSE_MASK_ID
from resolve.training.data import TrainingSentence, SenseData


def fix_mwe_discontinuity(training_sentence: TrainingSentence):
    sentence_mwe_ids = set(w.mwe_sense_data.item_id for w in training_sentence
                           if w.mwe_sense_data is not None) - {SENSE_MASK_ID}

    if sentence_mwe_ids != set(range(training_sentence.mwe_count)):
        # we deleted something from the middle, the range of MWE ids is no longer contiguous so we fix it
        sense_datas_by_id = defaultdict(list)
        for word in training_sentence:
            if word.mwe_sense_data is not None:
                sense_datas_by_id[word.mwe_sense_data.item_id].append(word.mwe_sense_data)

        for new_id, sense_data_list in enumerate(sense_datas_by_id.values()):
            for sense_data in sense_data_list:
                sense_data.item_id = new_id

@dataclass
class AnnotationApplicationStats:
    annotated: int = 0
    collisions: int = 0

    def __str__(self):
        return str(asdict(self))


def apply_mwe_annotations(sentences: List[TrainingSentence], annotation_list: List[MWEAnnotation]):
    stats = AnnotationApplicationStats()
    sentences_by_text = {
        sent.original_text: sent for sent in sentences
    }

    for annotation in tqdm(annotation_list, 'applying annotations'):
        sentence = sentences_by_text[annotation.sentence_text]
        target_words = [sentence[idx] for idx in annotation.word_indices]

        if any(w.mwe_sense_data is not None for w in target_words):
            stats.collisions += 1
            continue

        annotation.sense_data.item_id = annotation.annotation_id
        for word in target_words:
            word.mwe_sense_data = annotation.sense_data

        fix_mwe_discontinuity(sentence)
        stats.annotated += 1

    return stats


def filter_annotations(mwe_annotations: List[MWEAnnotation], max_mwe_count: Optional[int] = None,
                       additional_filter: Optional[Callable[[SenseData], bool]] = None) -> List[MWEAnnotation]:

    assert max_mwe_count is not None or additional_filter is not None

    if max_mwe_count is not None:
        mwe_counts = Counter()
        filtered_annotations = []
        for annotation in mwe_annotations:
            if mwe_counts[annotation.sense_data.lemma] < max_mwe_count:
                mwe_counts[annotation.sense_data.lemma] += 1
                filtered_annotations.append(annotation)
            else:
                continue

        mwe_annotations = filtered_annotations
        # shuffle since we delete more the further we get into the list
        random.Random(1337).shuffle(mwe_annotations)

    if additional_filter:
        mwe_annotations = [annot for annot in mwe_annotations if additional_filter(annot.sense_data)]

    return mwe_annotations


def mwe_one_stdev_max(sents: Iterable[TrainingSentence]) -> int:
    mwe_counts = Counter()
    for sentence in sents:
        for mwe_sense_data, mwe_words in sentence.get_mwe_groups(added_is=False):
            mwe_counts[mwe_sense_data.lemma] += 1

    return ceil(mean(mwe_counts.values()) + stdev(mwe_counts.values()))


class AnnotationStatus(Enum):
    PENDING = 'pending'
    DONE_HUMAN = 'done'
    DONE_AUTO = 'auto'


# used in prodigy output and when loading annotations to apply
ANNOTATION_KEY = '_annotation'


def compute_output_path(base: Path, addition: str) -> Path:
    assert addition not in base.name, f'Trying to add "{addition}" to file that already includes "{addition}'
    return base.parent / (('.'.join(base.name.split('.')[:-1])) + f'.{addition}.jsonl')


def read_annotations(annotation_file: Path) -> Iterable[MWEAnnotation]:
    str_path = str(annotation_file.absolute())
    with json_open(str_path) as json_file:
        for json_line in json_file:
            yield MWEAnnotation.from_json(json_line)


@dataclass(frozen=True)
class MWEAnnotation:
    sentence_text: str
    word_indices: Tuple[int]
    sense_data: SenseData
    status: AnnotationStatus

    def to_json(self) -> Dict:
        return {
            'sentence_text': self.sentence_text,
            'word_indices': list(self.word_indices),
            'sense_data': self.sense_data.to_json(),
            'status': self.status.value
        }

    @staticmethod
    def from_json(data: Dict) -> MWEAnnotation:
        return MWEAnnotation(
            data['sentence_text'],
            tuple(data['word_indices']),
            SenseData.from_json(data['sense_data'], None, True),
            AnnotationStatus(data['status'])
        )

    @property
    def annotation_id(self):
        return hash((self.sentence_text,) + self.word_indices)

    def __hash__(self):
        return self.annotation_id
