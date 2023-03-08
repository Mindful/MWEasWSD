from collections import namedtuple, defaultdict
from itertools import product, combinations
from typing import Optional, List, Union, Dict, Iterable, Callable

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

from resolve.common.data import BaseSentence, BaseWord
from resolve.mwe import MWECandidateList, MWEExtractionError
from resolve.mwe.detect import MWEDetector
from resolve.mwe.filter import MWEFilter, ModelOutputIsMWE
from resolve.mwe.index import MWEIndex, MWEData
from resolve.mwe.resolve import MWEResolver, KulkarniResolverWrapper
from resolve.model.pl_module import ContextDictionaryBiEncoder

DummyDefinitionData = namedtuple('DummyDefinitionData', ['is_mwe', 'mwe_score'])


class MWECandidate:

    def __init__(self, words: List[BaseWord], mwe_data: MWEData):
        self.words = words
        self.mwe_data = mwe_data
        self.definition_data = None

    def __repr__(self):
        return f'<MWECandidate for {self.mwe_data.lemma} from ({",".join(w.form for w in self.words)})>'

    @property
    def ordered(self) -> bool:
        return all(self.words[i].idx < self.words[i + 1].idx for i in range(len(self.words) - 1))

    @property
    def gappiness(self) -> int:
        """
        If the words are out of order in the sentence, distances may not be computed between the nearest two words
        so gappiness will be higher for out of order MWEs.
        """
        return sum(self.gaps) - (len(self.words) - 1)

    @property
    def gaps(self) -> List[int]:
        return [abs(self.words[i].idx - self.words[i + 1].idx) for i in range(len(self.words) - 1)]

    def collides_with(self, other: 'MWECandidate') -> bool:
        return any(w.idx in set(a.idx for a in self.words) for w in other.words)

    def set_definition_data(self, sentence: BaseSentence, model: ContextDictionaryBiEncoder,
                            tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], lang: str):

        definitions = self.mwe_data.get_definitions(lang, ignore_candidates=True)
        if definitions is not None:
            definition_batch = tokenizer(definitions, return_tensors='pt', padding=True)
            with torch.no_grad():
                self.definition_data = model.process_single_instance(sentence, self.words,
                                                                     definition_batch, mwe=True).cpu()
        else:
            # if we have no definitions, we just say it's an MWE (this should only happen with old wordnet data)
            # TODO: log warnings or handle this better somehow - this should only happen with Kulkarni
            self.definition_data = DummyDefinitionData(True, 1.0)


class PropNounMWECandidate(MWECandidate):

    def set_definition_data(self, sentence: BaseSentence, model: ContextDictionaryBiEncoder,
                            tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer], lang: str):
        """This says that proper noun MWEs are MWEs and of higher priority than any other"""
        self.definition_data = DummyDefinitionData(True, 1.0)

    @staticmethod
    def from_word_group(word_group: List[BaseWord]) -> 'PropNounMWECandidate':
        mwe_data = MWEData(
            lemma='_'.join(w.lemma.lower() for w in word_group),
            lemma_cased='_'.join(w.form for w in word_group),
            pos='propn',
            word_count=str(len(word_group))
        )
        return PropNounMWECandidate(word_group, mwe_data)


class PipelineOutput(MWECandidateList):
    def __init__(self, *args, **kwargs):
        if 'eval_data' in kwargs:
            self.eval_data = kwargs['eval_data']
        super(PipelineOutput, self).__init__(*args, **kwargs)


class MWEPipeline:
    def metadata(self):
        if self.resolver is None:
            resolver_name = 'None'
        else:
            resolver_name = self.resolver.__class__.__name__

        if isinstance(self.resolver, KulkarniResolverWrapper):
            resolver_name += f'({self.resolver._base_resolver.__class__.__name__})'

        return {
            'filters': [x.__class__.__name__ for x in self.filters],
            'resolver': resolver_name,
            'additional_detectors': [x.__class__.__name__ for x in self.additional_detectors],
            'index': self.index.index_path,
            'model': self.model.__class__.__name__ if self.model else 'None'
        }

    def __init__(self, index: MWEIndex, filters: List[MWEFilter], resolver: Optional[MWEResolver],
                 model: Optional[ContextDictionaryBiEncoder] = None, definition_lang: Optional[str] = None,
                 additional_detectors: Optional[List[MWEDetector]] = None, detect_with_form: bool = False,
                 compute_eval_stats: bool = False):
        self.index = index
        self.filters = filters
        self.resolver = resolver
        self.detect_with_form = detect_with_form
        self.compute_eval_stats = compute_eval_stats

        def get_with_form(indices: Dict[str, List[int]], w: BaseWord) -> int:
            index_list = indices.get(w.lemma, None)
            if not index_list:
                index_list = indices.get(w.form)

            if not index_list:
                raise MWEExtractionError('Should always be an index for either lemma or form')

            return index_list.pop(0)

        if self.detect_with_form:
            self.index_getter = get_with_form
        else:
            self.index_getter = lambda indices, w: indices[w.lemma].pop(0)

        self.additional_detectors = additional_detectors if additional_detectors else []
        self.model = model
        if self.model is not None:
            self.model.eval()
            definition_lang = definition_lang if definition_lang is not None else 'eng'

            self.context_tokenizer = AutoTokenizer.from_pretrained(model.context_encoder.name_or_path)
            self.model.set_context_tokenizer(self.context_tokenizer)
            if model.context_encoder.name_or_path == model.definition_encoder.name_or_path:
                self.definition_tokenizer = self.context_tokenizer
            else:
                self.definition_tokenizer = AutoTokenizer.from_pretrained(model.definition_encoder.name_or_path)

            self.definition_lang = definition_lang
        else:
            if definition_lang is not None:
                raise MWEExtractionError('Cannot set definition language on pipelines that do not include a model')

    def _detect(self, words: Iterable[BaseWord]) -> MWECandidateList:
        """Exhaustive detection"""
        all_lemmas = {w.lemma for w in words}
        if self.detect_with_form:
            non_lemmas = {w.form for w in words}
            all_lemmas = all_lemmas | non_lemmas
        possible_mwes = self.index.possible_mwes_for_lemmas(list(all_lemmas))

        lemma_to_words = {lemma: [] for mwe in possible_mwes for lemma in mwe.lemma_counter}
        for word in words:
            if word.lemma in lemma_to_words:
                lemma_to_words[word.lemma].append(word)
            if self.detect_with_form and word.form in lemma_to_words:
                lemma_to_words[word.form].append(word)

        output = []
        for mwe in possible_mwes:
            try:
                lemma_counter = mwe.lemma_counter

                # for each lemma, get the candidate words as lists (mostly of 1 item, but 2 for same lemma twice)
                candidate_word_combos = [list(combinations(lemma_to_words[lemma], lemma_counter[lemma]))
                                         for lemma in lemma_counter]

                if any(len(c) == 0 for c in candidate_word_combos):
                    continue  # this means we couldn't find enough of a specific lemma (I.E. we needed two but only had one)

                # compute all combinations of each of these lists, then flatten them
                mwe_combinations = {tuple(x for y in p for x in y) for p in product(*candidate_word_combos)}

                sorted_mwe_combinations = []

                for raw_combo in mwe_combinations:
                    indices = mwe.lemma_indices
                    appearance_order_combo = sorted(raw_combo, key=lambda w: w.idx)

                    final_order_combo = [None] * len(appearance_order_combo)
                    for w in appearance_order_combo:
                        final_order_combo[self.index_getter(indices, w)] = w

                    sorted_mwe_combinations.append(final_order_combo)

                output.extend(MWECandidate(combo, mwe) for combo in sorted_mwe_combinations)
            except MWEExtractionError as ex:
                #TODO: proper logging
                print(f'Extraction failed for MWE {mwe} on line {words} due to {ex}')

        return output

    def __call__(self, sent: BaseSentence,
                 candidate_word_filter: Optional[Callable[[BaseWord], bool]] = None) -> PipelineOutput:
        if candidate_word_filter is not None:
            candidate_words = [w for w in sent if candidate_word_filter(w)]
        else:
            candidate_words = sent

        eval_data = defaultdict(list)
        candidates = self._detect(candidate_words)
        for detector in self.additional_detectors:
            candidates.extend(detector(candidate_words))

        if self.model is not None:
            sent.set_input_ids(self.context_tokenizer)
            sent.set_embedding(self.model)

            for candidate in candidates:
                candidate.set_definition_data(sent, self.model, self.definition_tokenizer, self.definition_lang)

        filtered = candidates
        if self.compute_eval_stats and any(isinstance(f, ModelOutputIsMWE)
                                           for f in self.filters) and len(self.filters) > 1:
            # also compute what scores would look like if we only had the model filter (or if it was first)
            model_filter = next(f for f in self.filters if isinstance(f, ModelOutputIsMWE))
            eval_data['ModelFilterOnly'].append(model_filter.get_eval_data(filtered))

        for f in self.filters:
            if self.compute_eval_stats:
                eval_data[f.__class__.__name__].append(f.get_eval_data(filtered))

            filtered = f(filtered)

        if self.resolver is not None:
            final = self.resolver(filtered)
        else:
            # this may include overlapping MWEs since there's no resolver
            final = filtered

        return PipelineOutput(final, eval_data=eval_data if len(eval_data) > 0 else None)
