from __future__ import annotations

from abc import ABC
from typing import List, Dict, Union, Optional, Tuple, Set

import torch
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, BatchEncoding

SENSE_MASK_ID = -1

_lang2definition_linker = {
    'eng': ';',
    'jpn': '。'  # "。" is in some definitions, but that's probably fine
}
UPOS_TO_WN_POS_MAP = {'NOUN': wordnet.NOUN, 'PROPN': wordnet.NOUN, 'VERB': wordnet.VERB, 'AUX': wordnet.VERB,
                      'ADJ': wordnet.ADJ, 'ADV': wordnet.ADV, 'ADJ_SAT': wordnet.ADJ_SAT}


class WordnetDefinitionLookup:
    PARTS_OF_SPEECH = {wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADJ_SAT, wordnet.ADV}

    _candidate_synsets = None
    _synset_keys = None

    @classmethod
    def set_key_candidates(cls, key_candidates: Dict[Tuple[str, str], Set[str]]):
        print('Limiting possible wordnet results to those in key candidate dictionary')
        cls._candidate_synsets = {
            (lemma, pos): sorted([
                wordnet.lemma_from_key(key).synset() for key in keyset
            ]) for (lemma, pos), keyset in tqdm(key_candidates.items(), desc='build candidate dict')
        }
        cls._synset_keys = {
            (lemma, pos): {
                wordnet.lemma_from_key(key).synset(): key for key in keyset
            } for (lemma, pos), keyset in tqdm(key_candidates.items(), desc='build synset key dict')
        }

    @classmethod
    def candidates_fixed(cls) -> bool:
        return cls._candidate_synsets is not None

    @classmethod
    def key_for_synset(cls, lemma: str, pos: str, synset):
        assert cls._synset_keys is not None, 'Cannot fetch keys for synset unless candidate synsets are set'
        return cls._synset_keys[(lemma, pos)][synset]

    @classmethod
    def get_ordered_synsets(cls, lemma: str, pos: str, ignore_candidates: bool = False) -> List:
        if cls._candidate_synsets is not None and not ignore_candidates:
            return cls._candidate_synsets[(lemma, pos)]
        else:
            # sort this so it always come out in the same order (not clear if wordnet guarantees order)
            return sorted(wordnet.synsets(lemma, pos))

    @staticmethod
    def process_definition(definition: Union[str, List], lang: str) -> str:
        if isinstance(definition, list):
            linker = _lang2definition_linker.get(lang, None)
            assert linker is not None, f"Add a linker for this language: {lang}."

            return f'{linker} '.join(definition)
        else:
            return definition

    @classmethod
    def get_definitions(cls, lemma: str, pos: str, lang: str, ignore_candidates: bool = False) -> Optional[List[str]]:
        if pos not in cls.PARTS_OF_SPEECH:
            return None

        definitions = [synset.definition(lang=lang) for synset in cls.get_ordered_synsets(lemma, pos,
                                                                                          ignore_candidates)]
        if len(definitions) > 0:
            return [cls.process_definition(d, lang) for d in definitions if d is not None]
        else:
            return None


class BaseWord:
    def __init__(self, form: str, lemma: str):
        self.form = form
        self.lemma = lemma
        self.idx = None  # set by the sentence the word belongs to
        self.input_ids = None

    def __repr__(self):
        return f'<{self.form}>'

    def set_input_ids(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]):
        self.input_ids = tokenizer(self.form, add_special_tokens=False,
                                   return_tensors='pt')['input_ids'].squeeze(0).int()

    @property
    def word_id_tensor(self) -> torch.Tensor:
        return torch.full(self.input_ids.shape, self.idx).int()


class BaseSentence:
    def __init__(self, words: List[BaseWord], original_text: str):
        for idx, word in enumerate(words):
            word.idx = idx

        self.words = words
        self.original_text = original_text
        self.input_ids = None
        self.embedding = None
        self.token_word_ids = None

    def __iter__(self):
        return iter(self.words)

    def __getitem__(self, idx: int) -> BaseWord:
        return self.words[idx]

    def __repr__(self):
        return f'<{self.original_text}>'

    def __len__(self):
        return len(self.words)

    def __hash__(self):
        return hash(self.original_text)

    def set_input_ids(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]):
        for word in self:
            if word.input_ids is None:
                word.set_input_ids(tokenizer)

        self.input_ids = torch.cat([torch.tensor([tokenizer.cls_token_id])] +
                                   [w.input_ids for w in self] +
                                   [torch.tensor([tokenizer.sep_token_id])])

        token_word_ids = [w.word_id_tensor for w in self]
        token_word_ids = [torch.tensor([SENSE_MASK_ID])] + token_word_ids + [torch.tensor([SENSE_MASK_ID])]
        self.token_word_ids = torch.cat(token_word_ids)

    def set_embedding(self, model: 'ContextDictionaryBiEncoder'):
        with torch.no_grad():
            # unsqueeze 0th dimension so the input has a batch dimension
            self.embedding = model.context_forward(self.input_ids.unsqueeze(0).to(model.device)).squeeze().cpu()
