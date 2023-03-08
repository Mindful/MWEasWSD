from abc import ABC
from zlib import adler32

from resolve.mwe import MWECandidateList, MWEExtractionError


class MWEResolver(ABC):
    def __call__(self, candidates: MWECandidateList) -> MWECandidateList:
        ordered_candidates = sorted(candidates, key=self._score, reverse=True)
        i = 0
        while i < len(ordered_candidates):
            current = ordered_candidates[i]
            ordered_candidates = [c for c in ordered_candidates
                                  if c == current or not c.collides_with(current)]
            i += 1

        return ordered_candidates

    def _score(self, candidate: 'MWECandidate'):
        raise NotImplementedError()


class NaiveResolver(MWEResolver):

    def _score(self, candidate: 'MWECandidate'):
        # deterministic hashing function so order is random but reproduceable
        return adler32(candidate.mwe_data.lemma_cased.encode())


class LessGappyLongerResolver(MWEResolver):
    def _score(self, candidate: 'MWECandidate'):
        return -candidate.gappiness, len(candidate.words)


class KulkarniResolverWrapper(MWEResolver):
    def __init__(self, base_resolver: MWEResolver):
        self._base_resolver = base_resolver

    def _score(self, candidate: 'MWECandidate'):
        from resolve.mwe.pipeline import PropNounMWECandidate

        # hack to prioritize prop noun candidates
        if isinstance(candidate, PropNounMWECandidate):
            return 1000,
        else:
            return self._base_resolver._score(candidate)


class ModelScoreResolver(MWEResolver):
    def _score(self, candidate: 'MWECandidate'):
        if candidate.definition_data is None:
            raise MWEExtractionError('Cannot use model based resolver in a pipeline with no model')

        return candidate.definition_data.mwe_score,



