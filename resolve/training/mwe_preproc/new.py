import typing
from dataclasses import dataclass, asdict, field
from collections import Counter
from typing import Optional, List, Iterable, Tuple

from resolve.common.data import SENSE_MASK_ID
from resolve.mwe.pipeline import MWEPipeline, PropNounMWECandidate
from resolve.training.data import TrainingSentence, SenseData
from resolve.training.mwe_preproc.common import AnnotationStatus, MWEAnnotation


@dataclass
class NewMWEStats:
    gappiness: typing.Counter = field(default_factory=Counter)
    pos: typing.Counter = field(default_factory=Counter)

    def __str__(self):
        return str(asdict(self))


def all_new_candidates(pipeline: MWEPipeline,
                       sentences: Iterable[TrainingSentence]) -> Tuple[NewMWEStats, List[MWEAnnotation]]:
    stats = NewMWEStats()
    all_candidates = []
    for sent in sentences:
        all_candidates.extend(get_new_candidates(pipeline, sent, stats))

    return stats, all_candidates


def get_new_candidates(pipeline: MWEPipeline, sentence: TrainingSentence,
                       stats: Optional[NewMWEStats]) -> List[MWEAnnotation]:
    if stats is None:
        stats = NewMWEStats()

    candidates = pipeline(sentence, candidate_word_filter=lambda w: w.mwe_sense_data is None)
    annotation_candidates = []

    for candidate in candidates:
        if isinstance(candidate, PropNounMWECandidate):
            # MWE candidates don't correspond to things we can look up definitions for
            # so we don't use them
            continue

        new_sense_data = SenseData(
                candidate.mwe_data.lemma_cased,
                candidate.mwe_data.pos,
                SENSE_MASK_ID,
                None,
                None,
                True,
                {
                    'added': True
                }
            )

        annotation_candidates.append(
            MWEAnnotation(
                sentence_text=sentence.original_text,
                word_indices=tuple(w.idx for w in candidate.words),
                sense_data=new_sense_data,
                status=AnnotationStatus.PENDING
            )
        )

        if candidate.gappiness < 10:
            stats.gappiness[candidate.gappiness] += 1
        else:
            stats.gappiness['>=10'] += 1
        stats.pos[candidate.mwe_data.pos] += 1

    return annotation_candidates
