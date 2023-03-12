from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Set, List

from resolve.mwe import MWECandidateList, MWEExtractionError


@dataclass
class FilterEvalData:
    word_hyp_labels: List[int]
    word_gold_labels: List[int]
    mwe_hyp_labels: List[int]
    mwe_gold_labels: List[int]


class MWEFilter(ABC):
    def __call__(self, candidates: MWECandidateList) -> MWECandidateList:
        return [x for x in candidates if self.acceptable(x)]

    def get_eval_data(self, candidates: MWECandidateList):
        all_word_hyp_labels = []
        all_word_gold_labels = []
        all_mwe_hyp_labels = []
        all_mwe_gold_labels = []

        for candidate in candidates:
            word_gold_labels = [1 if w.mwe_sense_data is not None else 0 for w in candidate.words]
            gold_label_set = set(word_gold_labels)
            # if labels are mixed, 2 for partial match
            mwe_gold_label = 2 if len(gold_label_set) > 1 else word_gold_labels[0]

            hyp_label = 1 if self.acceptable(candidate) else 0
            word_hyp_labels = [hyp_label] * len(word_gold_labels)

            all_word_hyp_labels.extend(word_hyp_labels)
            all_word_gold_labels.extend(word_gold_labels)
            all_mwe_hyp_labels.append(hyp_label)
            all_mwe_gold_labels.append(mwe_gold_label)

        return FilterEvalData(all_word_hyp_labels, all_word_gold_labels, all_mwe_hyp_labels, all_mwe_gold_labels)

    @abstractmethod
    def acceptable(self, x: 'MWECandidate'):
        raise NotImplementedError()


class OrderedOnly(MWEFilter):
    def acceptable(self, x: 'MWECandidate'):
        return x.ordered


class MaxGappiness(MWEFilter):
    def __init__(self, max_gappiness: int):
        self.max_gappiness = max_gappiness

    def acceptable(self, x: 'MWECandidate'):
        return x.gappiness <= self.max_gappiness


class SpecificPOSOnly(MWEFilter):
    def __init__(self, pos_set: Set[str]):
        self.pos_set = pos_set

    def acceptable(self, x: 'MWECandidate'):
        return x.mwe_data.pos in self.pos_set


class InvertedFilter(MWEFilter):
    def __init__(self, original_filter: MWEFilter):
        self.original = original_filter

    def acceptable(self, x: 'MWECandidate'):
        return not self.original.acceptable(x)


class UnionFilter(MWEFilter):
    def __init__(self, *filters: MWEFilter):
        self.filters = filters

    def acceptable(self, x: 'MWECandidate'):
        return any(f.acceptable(x) for f in self.filters)


class ModelOutputIsMWE(MWEFilter):
    def acceptable(self, x: 'MWECandidate'):
        if x.definition_data is None:
            raise MWEExtractionError('Cannot use model based filter in a pipeline with no model')

        return x.definition_data.is_mwe


class AllowedVocabFilter(MWEFilter):
    """Used primarily to limit output vocabulary for pipeline comparison/evaluation"""

    def __init__(self, vocab_path: Path):
        with vocab_path.open('r') as vocab_file:
            self.vocab = set(mwe_line.strip() for mwe_line in vocab_file)

    def acceptable(self, x: 'MWECandidate'):
        return x.mwe_data.lemma in self.vocab
