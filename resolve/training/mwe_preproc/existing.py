from __future__ import annotations

import re
import typing
from collections import namedtuple, Counter
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from typing import List, Optional, Iterable

from nltk import WordNetLemmatizer

from resolve.common.data import UPOS_TO_WN_POS_MAP, SENSE_MASK_ID
from resolve.training.data import TrainingWord, TrainingSentence

ElementResult = namedtuple('ElementResult',
                           ['mwe_labels', 'label_lemmas', 'found_forms', 'found_lemmas', 'incomplete', 'mismatch',
                            'idx'])

lemmatizer = WordNetLemmatizer()
original_pos_map = {value: key for key, value in UPOS_TO_WN_POS_MAP.items()}


class MWECandidateType(Enum):
    FOUND = auto()
    NOT_FOUND = auto()
    MULTIPLE_CANDIDATES = auto()


mwe_word_split_regex = re.compile(r'[-\s]')


@dataclass
class ExistingMWEStats:
    incomplete: int = 0
    unfound_remainder: int = 0
    realigned: int = 0
    multiple_candidates: int = 0
    mismatch: int = 0
    complete: int = 0
    mwe_counts: typing.Counter = field(default_factory=Counter)
    pos_counts: typing.Counter = field(default_factory=Counter)

    def __str__(self):
        return str(asdict(self))


def _lemmatize(form: str, input_pos: str) -> str:
    pos = original_pos_map[input_pos]
    if form == 'lay' and pos == 'VERB':
        return 'lie'

    if pos in UPOS_TO_WN_POS_MAP:
        return lemmatizer.lemmatize(form, UPOS_TO_WN_POS_MAP[pos])
    else:
        return lemmatizer.lemmatize(form)


def _assign_item_ids(sentence: TrainingSentence):
    mwe_count = 0
    existing_mwe_word_ids = {}
    for word in sentence:
        if word.mwe_sense_data is not None:
            mwe_id = word.mwe_sense_data.item_id
            assert mwe_id != SENSE_MASK_ID
            # use parent mwe data if this word is part of a realigned MWE
            mwe_word_idx = word.mwe_sense_data.metadata.get('parent_mwe_word_sense',
                                                            word.word_sense_data.item_id)
            if mwe_word_idx not in existing_mwe_word_ids:
                existing_mwe_word_ids[mwe_word_idx] = mwe_count
                mwe_count += 1

            word.mwe_sense_data.item_id = existing_mwe_word_ids[mwe_word_idx]


def find_existing_mwe(words: List[TrainingWord], idx: int) -> Optional[ElementResult]:
    head_word = words[0]
    assert len(set(word.mwe_sense_data for word in words)) == 1, \
        f'Grouped words must share MWE sense but found {words} with {set(w.mwe_sense_data for w in words)}'

    if head_word.mwe_sense_data is None or 'parent_mwe_word_sense' in head_word.mwe_sense_data.metadata:
        return None

    mwe_labels = head_word.mwe_sense_data.lemma.split('_')
    text = ' '.join(w.form for w in words)

    found_forms = mwe_word_split_regex.split(text)
    if len(found_forms) > len(mwe_labels):
        # this means we likely split on a dash where we shouldn't have
        found_forms = text.split()

    # try this without lemmatization first, lemmatizer is weird sometimes
    found_lemmas = [form.lower() for form in found_forms]
    label_lemmas = [form.lower() for form in mwe_labels]
    if found_lemmas == label_lemmas or len(set(found_lemmas) & set(label_lemmas)) == len(found_lemmas):
        lemma_overlap = len(found_lemmas)
    else:
        found_lemmas = [_lemmatize(form, head_word.mwe_sense_data.pos).lower() for form in found_forms]
        label_lemmas = [_lemmatize(form, head_word.mwe_sense_data.pos).lower() for form in mwe_labels]

        # this disregards order but most MWEs don't contain the same lemma multiple times, so it should be fine
        lemma_overlap = len(set(found_lemmas) & set(label_lemmas))

    mismatch = lemma_overlap != len(label_lemmas) and len(found_lemmas) >= len(label_lemmas)

    incomplete = len(mwe_labels) > len(found_forms) and not ''.join(found_lemmas) == ''.join(label_lemmas)

    return ElementResult(mwe_labels, label_lemmas, found_forms, found_lemmas, incomplete, mismatch, idx)


def process(sentences: Iterable[TrainingSentence]) -> ExistingMWEStats:
    stats = ExistingMWEStats()
    for sent in sentences:
        process_existing(sent, stats)

    return stats


def process_existing(sentence: TrainingSentence, stats: Optional[ExistingMWEStats]):
    if stats is None:
        stats = ExistingMWEStats()

    sentence_mwe_count = 0

    words_by_word_id = {
        idx: [w for w in sentence if w.word_sense_data.item_id == idx]
        for idx in set(w.word_sense_data.item_id for w in sentence)
    }

    need_to_assign_item_ids = False
    for idx, words in words_by_word_id.items():
        head_word = words[0]
        element_result = find_existing_mwe(words, sentence_mwe_count)
        if element_result is not None:
            sentence_mwe_count += 1
            stats.pos_counts[head_word.mwe_sense_data.pos] += 1

            if element_result.mismatch:
                stats.mismatch += 1
                continue

            if element_result.incomplete:
                stats.incomplete += 1

                missing_lemmas = [lemma for lemma in element_result.label_lemmas
                                  if lemma not in element_result.found_lemmas]

                min_lemmas_to_realign = min(2, len(missing_lemmas))
                candidate_type_tuples = []

                for lemma in missing_lemmas:
                    candidates = [w for w in sentence
                                  # word is the right form/lemma (I.E. right word)
                                  if (w.word_sense_data.lemma == lemma or w.form.lower() == lemma)
                                  # other word not a MWE, or was originally the same word
                                  and (w.mwe_sense_data is None or w.word_sense_data.item_id ==
                                       head_word.word_sense_data.item_id)
                                  # word comes after head word; MWEs have to be in order
                                  and w.idx > head_word.idx]
                    if len(candidates) == 1:
                        candidate_type_tuples.append((MWECandidateType.FOUND, candidates))
                    elif len(candidates) == 0:
                        # if 0, we couldn't find the rest of the MWE, which is weird
                        candidate_type_tuples.append((MWECandidateType.NOT_FOUND, candidates))
                    else:
                        # if > 1, there are multiple candidates and we don't know which one to use so we pick none
                        candidate_type_tuples.append((MWECandidateType.MULTIPLE_CANDIDATES, candidates))

                type_counts = Counter(t for t, _ in candidate_type_tuples)
                if type_counts[MWECandidateType.FOUND] >= min_lemmas_to_realign:
                    for t, candidates in candidate_type_tuples:
                        if t == MWECandidateType.FOUND:
                            candidate = candidates[0]
                            candidate.mwe_sense_data = deepcopy(head_word.mwe_sense_data)
                            candidate.mwe_sense_data.metadata[
                                'parent_mwe_word_sense'] = head_word.word_sense_data.item_id

                    stats.realigned += 1
                    need_to_assign_item_ids = True
                    for w in words:
                        w.mwe_sense_data.metadata['type'] = 'realigned'

                elif type_counts[MWECandidateType.NOT_FOUND] >= type_counts[MWECandidateType.MULTIPLE_CANDIDATES]:
                    stats.unfound_remainder += 1
                else:
                    stats.multiple_candidates += 1
            else:
                stats.complete += 1

    stats.mwe_counts[sentence_mwe_count] += 1
    if need_to_assign_item_ids:
        _assign_item_ids(sentence)  # necessary to updates for parent senses, even with no new additions

