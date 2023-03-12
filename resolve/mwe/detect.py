from abc import ABC
from typing import Iterable

from resolve.common.data import BaseSentence
from resolve.mwe import MWECandidateList
from resolve.training.data import TrainingWord


class MWEDetector(ABC):
    def __call__(self, sent: BaseSentence) -> MWECandidateList:
        raise NotImplementedError()


class DimSumSequentialNounDetector(MWEDetector):
    """Only ued for dimsum; marks consecutive nouns as MWE"""
    def __call__(self, words: Iterable[TrainingWord]) -> MWECandidateList:
        from resolve.mwe.pipeline import PropNounMWECandidate

        noun_words = [w for w in words if w.word_sense_data.metadata['pos'] in {'NOUN', 'PROPN'}]
        if len(noun_words) == 0:
            return []

        last = noun_words[0]
        cur_group = [noun_words[0]]
        word_groups = []
        for word in noun_words[1:]:
            if word.idx - 1 == last.idx:  # Part of the group, bump the end
                cur_group.append(word)
                last = word
            else:  # Not part of the group, yield current group and start a new
                word_groups.append(cur_group)

                last = word
                cur_group = [word]

        word_groups.append(cur_group)

        return [
            PropNounMWECandidate.from_word_group(word_group)
            for word_group in word_groups
            if len(word_group) > 1
        ]


class KulkarniPropNounDetector(MWEDetector):
    """Only used for detecting proper nouns in the Kulkarni data"""

    def __call__(self, words: Iterable[TrainingWord]) -> MWECandidateList:
        from resolve.mwe.pipeline import PropNounMWECandidate

        propn_words = [w for w in words if w.word_sense_data.pos == 'NNP'
                       or w.word_sense_data.pos == 'NNPS']
        if len(propn_words) == 0:
            return []

        last = propn_words[0]
        cur_group = [propn_words[0]]
        word_groups = []
        for word in propn_words[1:]:
            if word.idx - 1 == last.idx:  # Part of the group, bump the end
                cur_group.append(word)
                last = word
            else:  # Not part of the group, yield current group and start a new
                word_groups.append(cur_group)

                last = word
                cur_group = [word]

        word_groups.append(cur_group)

        return [
            PropNounMWECandidate.from_word_group(word_group)
            for word_group in word_groups
            if len(word_group) > 1
        ]


