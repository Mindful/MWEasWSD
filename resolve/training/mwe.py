from __future__ import annotations

import re
from collections import defaultdict
from functools import reduce
from typing import List, Optional

from resolve.training.data import TrainingWord, TrainingSentence, TrainingDefinitionManager, \
    SenseData, Detokenizer
from resolve.common.data import UPOS_TO_WN_POS_MAP


class KulkarniWord(TrainingWord):

    def __init__(self, word_string: str, manager: Optional[TrainingDefinitionManager]):

        parts = [x for x in word_string.split('_') if x != '']
        if len(parts) == 4:  # no lemma:
            form, pos, word_id, word_sub_id = parts
            lemma = form.lower()
        elif len(parts) >= 5:
            form, pos, lemma = parts[:3]
            word_id, word_sub_id = parts[-2:]
        else:
            raise RuntimeError(f'{word_string} split into an unsupported number of parts')

        self.token_id = f'{word_id}_{word_sub_id}'
        word_sense_data = SenseData(lemma, pos, int(word_id), None, manager, False)
        mwe_sense_data = None

        super(KulkarniWord, self).__init__(form, word_sense_data, mwe_sense_data, manager)


class KulkarniSentence(TrainingSentence):
    split_regex = re.compile(r'(?<=\d),(?!_)')

    def __init__(self, sentence_text: str, mwe_lines: List[str], manager: Optional[TrainingDefinitionManager]):
        sentence_tokens = sentence_text.split()
        self.sent_id = sentence_tokens[0]
        words = [KulkarniWord(wstr, manager) for wstr in sentence_tokens[1:]]

        words_by_token_id = {
            w.token_id: w for w in words
        }

        original_text = Detokenizer.detokenize(words)

        for idx, mwe_line in enumerate(mwe_lines):
            mwe_id, mwe_word_ids = mwe_line.split('={')
            lemma_end_index = mwe_id.rfind('_')
            mwe_lemma = mwe_id[:lemma_end_index]
            pos = mwe_id[lemma_end_index + 1:]

            mwe_word_ids = self.split_regex.split(mwe_word_ids[:-1])  # remove final "}"
            for mwe_word_id in mwe_word_ids:
                id_data = mwe_word_id.split('_')
                word_id, word_sub_id = id_data[-2:]
                form = id_data[0]
                token_id = f'{word_id}_{word_sub_id}'
                target_word = words_by_token_id[token_id]
                assert target_word.form == form
                target_word.mwe_sense_data = SenseData(mwe_lemma, pos, idx, None, manager, True)

        super(KulkarniSentence, self).__init__(words, original_text, manager)


class DimSumWord(TrainingWord):
    def __init__(self, idx: str, word: str, lemma: str, pos: str, mwe_tag: str, parent_idx: str, unused: str,
                 supersense: str, sent_id: str, manager: Optional[TrainingDefinitionManager]):
        original_data = {k: v for k, v in locals().items() if isinstance(v, str)}

        pos = UPOS_TO_WN_POS_MAP[pos] if pos in UPOS_TO_WN_POS_MAP else pos
        word_sense_data = SenseData(lemma, pos, int(idx) - 1, None, manager, False, metadata=original_data)
        # we populate the MWE sense data in the sentence postprocessing

        super().__init__(word, word_sense_data, None, manager)


class DimSumSentence(TrainingSentence):
    def __init__(self, words: List[TrainingWord], manager: Optional[TrainingDefinitionManager]):
        original_text = Detokenizer.detokenize(words)

        mwe_inners = [word for word in words if word.word_sense_data.metadata['mwe_tag'] == 'I']
        inner_parents = {word.word_sense_data.metadata['parent_idx'] for word in mwe_inners}

        mwe_tails = [word for word in mwe_inners if word.word_sense_data.metadata['idx']
                     not in inner_parents]

        mwe_groups = []
        for tail in mwe_tails:
            group = []
            current = tail
            while current.word_sense_data.metadata['mwe_tag'] == 'I':
                group.append(current)
                current = next(word for word in words
                               if word.word_sense_data.metadata['idx'] ==
                               current.word_sense_data.metadata['parent_idx'])

            assert current.word_sense_data.metadata['mwe_tag'] == 'B'
            group.append(current)
            group.reverse()
            mwe_groups.append(group)

        for idx, mwe_group in enumerate(mwe_groups):
            lemma = '_'.join(w.lemma for w in mwe_group)
            pos = mwe_group[0].word_sense_data.pos
            sense_data = SenseData(lemma, pos, idx, None, manager, True)
            for word in mwe_group:
                word.mwe_sense_data = sense_data

        super().__init__(words, original_text, manager,
                         metadata={'id': words[0].word_sense_data.metadata['sent_id']})


class ConnlUWord(TrainingWord):
    # for cupt: # global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE
    # for STREUSLE https://github.com/nert-nlp/streusle/blob/master/CONLLULEX.md#body

    digit_regex = re.compile(r'\d+')
    blank_mwe_strings = {'_', '*'}

    def __init__(self, id: str, form: str, lemma: str, upos: str, xpos: str, feats: str, head: str, deprel: str,
                 deps: str, misc: str, mwe: str,  # attributes present in both CUPT and STREUSLE end here
                 lexcat: Optional[str] = None, lexlemma: Optional[str] = None, supersense: Optional[str] = None,
                 second_supersense: Optional[str] = None, weak_mwe: Optional[str] = None,
                 wcat_unused: Optional[str] = None, weakmwe_lemma: Optional[str] = None, bio_tag: Optional[str] = None,
                 manager: Optional[TrainingDefinitionManager] = None):

        original_data = {k: v for k, v in locals().items() if isinstance(v, str)}

        pos = UPOS_TO_WN_POS_MAP[upos] if upos in UPOS_TO_WN_POS_MAP else upos

        mwe_str = ';'.join((x for x in (mwe, weak_mwe) if x is not None and x not in self.blank_mwe_strings))

        word_sense_data = SenseData(lemma, pos, int(id.split('.')[0]) - 1, None, manager, False, metadata=original_data)
        if mwe_str == '':
            mwe_sense_data = None
            self.ignored_mwe_ids = set()
        else:
            # for CUPT, MWEs are "id:type" with possible semicolon for overlap. like "1:LVC.full;2:IAV"
            # for STREUSLE, MWEs are "id:token_id" but we can have both mwe and weak mwe, and in that case
            # we join them so "2:1" and "1:2" -> "2:1;1:2".
            # in all cases, we take the first digit as the mwe ID. for cupt this is the first MWE listed,
            # for STREUSLE it is the strong MWE id if present, otherwise the weak MWE id
            mwe_ids = sorted((m.group() for m in self.digit_regex.finditer(mwe_str)), key=lambda d: int(d))
            mwe_id = mwe_ids[0]  # only the first digit, exclude type data and other IDs
            self.ignored_mwe_ids = set(int(i) - 1 for i in mwe_ids[1:])
            mwe_sense_data = SenseData(mwe_id, pos, int(mwe_id) - 1, None, manager, True)

        super(ConnlUWord, self).__init__(form, word_sense_data, mwe_sense_data, manager)


class ConnlUSentence(TrainingSentence):

    def __init__(self, words: List[ConnlUWord], sent_id: str, original_text: str,
                 manager: Optional[TrainingDefinitionManager]):
        mwe_lemmas = defaultdict(list)
        for word in words:
            if word.mwe_sense_data is not None:
                mwe_lemmas[word.mwe_sense_data.item_id].append(word.word_sense_data.lemma)

        final_mwe_lemmas = {
            mwe_id: '_'.join(mwe_lemmas) for mwe_id, mwe_lemmas in mwe_lemmas.items()
        }
        ignored_mwe_ids = reduce(lambda total, current: total | current, (w.ignored_mwe_ids for w in words))

        for word in words:
            if word.mwe_sense_data is not None:
                if word.mwe_sense_data.item_id in ignored_mwe_ids:
                    word.mwe_sense_data = None
                else:
                    word.mwe_sense_data.lemma = final_mwe_lemmas[word.mwe_sense_data.item_id]

        super(ConnlUSentence, self).__init__(words, original_text, manager, metadata={'id': sent_id})
