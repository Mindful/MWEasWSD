from __future__ import annotations

import dataclasses
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import takewhile, repeat
from pathlib import Path
from pprint import pp
from typing import Optional, List, Union, Any, Dict, Callable, Iterator, Type, Tuple, Iterable, Set

import pytorch_lightning as pl
import torch
from jsonlines import open as json_open
from nltk import TreebankWordDetokenizer
from nltk.corpus import wordnet
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, BatchEncoding

from resolve.common.data import BaseSentence, BaseWord, SENSE_MASK_ID, WordnetDefinitionLookup
from resolve.training import WSD_PATH

NEGATIVE_MWE_GOLD_SENSE = '<NOT_AN_MWE>'


# https://stackoverflow.com/a/27518377/4243650
def fast_linecount(filename: Union[str, Path]) -> int:
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        linecount = sum(buf.count(b'\n') for buf in bufgen)

    return linecount


def wsd_candidates(file_path: Path) -> Dict[Tuple[str, str], Set[str]]:
    candidates = {}
    with file_path.open('r') as input_file:
        for line in input_file:
            values = line.split('\t')
            lemma = values[0]
            pos = values[1]
            keys = set(x.strip() for x in values[2:])

            candidates[(lemma, pos)] = keys

    return candidates


def use_only_candidate_wordnet(fallback: bool = True):
    candidates_path = WSD_PATH / 'Data_Validation' / 'candidatesWN30.txt'
    candidate_set = wsd_candidates(candidates_path)
    WordnetDefinitionLookup.set_key_candidates(candidate_set)


def read_training_sentences(file_path: Path, manager: Optional[TrainingDefinitionManager],
                            disable_tqdm: bool = False) -> Iterator[TrainingSentence]:
    str_path = str(file_path.absolute())
    line_count = fast_linecount(str_path)
    for json_line in tqdm(json_open(str_path), total=line_count, desc=file_path.name, disable=disable_tqdm):
        try:
            yield TrainingSentence.from_json(json_line, manager)
        except Exception as ex:
            print('Failed to load below json as a training sentence:')
            print(json_line)
            raise ex


class SenseData:
    __slots__ = ['key_id', 'gold_sense_idx', 'ignored', 'lemma', 'pos', 'item_id', 'gold_sense', '_metadata']

    @property
    def is_mwe(self) -> bool:
        return '_' in self.lemma

    def __repr__(self):
        return f'<SenseData {self.item_id} for ({self.lemma}, {self.pos})>'

    def __hash__(self):
        return hash(tuple(getattr(self, a) for a in self.__slots__ if not a.startswith('_')))

    def __eq__(self, other):
        if not isinstance(other, SenseData):
            return False

        for attr in self.__slots__:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def force_ignore(self):
        self.ignored = True
        self.gold_sense_idx = SENSE_MASK_ID
        self.key_id = SENSE_MASK_ID

    @property
    def added_but_not_annotated(self) -> bool:
        return self.metadata.get('added', False) and not self.metadata.get('annotated', False)

    @property
    def metadata(self) -> Dict:
        if self._metadata is None:
            self._metadata = {}

        return self._metadata

    def get_definitions(self, lang: str, ignore_candidates: bool = False) -> Optional[List[str]]:
        return WordnetDefinitionLookup.get_definitions(self.lemma, self.pos, lang, ignore_candidates)

    def __init__(self, lemma: str, pos: str, item_id: int, gold_sense: Optional[str],
                 manager: Optional[TrainingDefinitionManager], allow_single_def: bool, metadata: Optional[Dict] = None):
        assert lemma
        assert pos
        self._metadata = metadata
        self.lemma = lemma
        self.pos = pos
        self.item_id = item_id
        self.gold_sense_idx = SENSE_MASK_ID
        self.key_id = SENSE_MASK_ID
        self.gold_sense = gold_sense

        if gold_sense is not None and manager is not None:
            # Annotations were done with all the wordnet 3.0 data, so handle those differently
            # (since senses or entire MWEs might be missing in the old candidate data)
            annotated = self.metadata.get('annotated', False)
            definitions = self.get_definitions(manager.def_language, ignore_candidates=annotated)
            assert definitions  # definitions should be present if this is a training example
            self.ignored = len(definitions) <= 1 and not allow_single_def

            if not self.ignored:
                key = 'ann:' + self.key if annotated else self.key
                self.key_id = manager.register_word_key(key, definitions)

                if gold_sense == NEGATIVE_MWE_GOLD_SENSE:
                    self.gold_sense_idx = len(definitions)  # "not mwe" tensor is last tensor
                else:
                    gold_definition = WordnetDefinitionLookup.process_definition(
                        wordnet.lemma_from_key(gold_sense).synset().definition(lang=manager.def_language),
                        manager.def_language)
                    self.gold_sense_idx = definitions.index(gold_definition)
                    assert self.gold_sense_idx != -1, f'Gold sense must be present in known definitions for {self}'
        else:
            self.ignored = False

    @property
    def is_mwe_negative(self) -> bool:
        return self.gold_sense == NEGATIVE_MWE_GOLD_SENSE

    def to_json(self) -> Dict:
        return {
            'lemma': self.lemma,
            'pos': self.pos,
            'idx': self.item_id,
            'gold_sense': self.gold_sense,
            'meta': self._metadata if self._metadata else None
        }

    @staticmethod
    def from_json(json: Dict, manager: Optional[TrainingDefinitionManager], allow_single_def: bool) -> SenseData:
        return SenseData(json['lemma'], json['pos'], json['idx'], json.get('gold_sense', None),
                         manager, allow_single_def, metadata=json.get('meta', None))

    @property
    def labeled(self) -> bool:
        not_masked = not any(getattr(self, x) == SENSE_MASK_ID for x in ('key_id', 'gold_sense_idx', 'item_id'))
        return not_masked and not self.ignored

    @property
    def key(self) -> str:
        return f'{self.lemma}+{self.pos}'

    def label_to_key(self, label: int, fallback: bool = False) -> str:
        ordered_synsets = WordnetDefinitionLookup.get_ordered_synsets(self.lemma, self.pos)
        if label == len(ordered_synsets):
            return NEGATIVE_MWE_GOLD_SENSE
        synset = ordered_synsets[label]

        if WordnetDefinitionLookup.candidates_fixed():
            return WordnetDefinitionLookup.key_for_synset(self.lemma, self.pos, synset)

        lemma_candidates = [
            lemma for lemma in synset.lemmas() if lemma.name().lower() == self.lemma.lower()
        ]
        if fallback and len(lemma_candidates) == 0:
            lemma_candidates = [
                lemma for lemma in synset.lemmas() if lemma.name().lower() in self.lemma.lower()
        ]
        assert len(lemma_candidates) == 1, f'Lemma for synset corresponding to label must be unambiguous, ' \
                                           f'found {len(lemma_candidates)} for {(self.lemma, self.pos)}'
        return lemma_candidates[0].key()


class TrainingWord(BaseWord):

    def __repr__(self):
        return f'<{self.form}>'

    def __init__(self, form: str, word_sense_data: SenseData, mwe_sense_data: Optional[SenseData],
                 manager: Optional[TrainingDefinitionManager]):
        self.word_sense_data = word_sense_data
        self.mwe_sense_data = mwe_sense_data
        super(TrainingWord, self).__init__(form, self.word_sense_data.lemma)

        if manager is not None:
            self.set_input_ids(manager.context_tokenizer)

    def to_json(self) -> Dict:
        return {
            'form': self.form,
            'word': self.word_sense_data.to_json(),
            'mwe': self.mwe_sense_data.to_json() if self.mwe_sense_data is not None else None
        }

    @staticmethod
    def from_json(json: Dict, manager: Optional[TrainingDefinitionManager]) -> TrainingWord:
        word_allow_single_def = manager.allow_single_def if manager is not None else False
        word_sense_data = SenseData.from_json(json['word'], manager, word_allow_single_def)
        mwe_sense_data = SenseData.from_json(json['mwe'], manager, True) if json['mwe'] is not None else None

        return TrainingWord(json['form'], word_sense_data, mwe_sense_data, manager)

    @property
    def word_id_tensor(self) -> torch.Tensor:
        return torch.full(self.input_ids.shape, self.word_sense_data.item_id)

    @property
    def mwe_id_tensor(self) -> torch.Tensor:
        mwe_id = self.mwe_sense_data.item_id if self.mwe_sense_data is not None else SENSE_MASK_ID
        return torch.full(self.input_ids.shape, mwe_id)


class TrainingSentence(BaseSentence):

    def __init__(self, words: List[TrainingWord], original_text: str, manager: Optional[TrainingDefinitionManager],
                 metadata: Optional[Dict] = None):
        super(TrainingSentence, self).__init__(words, original_text)

        self._metadata = metadata
        if manager is None:
            return

        assert all(word.mwe_sense_data is None or not word.mwe_sense_data.added_but_not_annotated
                   for word in self), f'Sentence {self} contains automatically added MWE data that is not annotated'

        if manager.mwe and manager.wsd:
            # ignore word sense data that is identical to the mwe sense data
            for word in words:
                if word.mwe_sense_data and word.word_sense_data.key_id == word.mwe_sense_data.key_id != SENSE_MASK_ID:
                    word.word_sense_data.force_ignore()

        self.set_input_ids(manager.context_tokenizer)
        assert self.input_ids.shape == self.token_word_ids.shape, 'Must be a word ID for every input ID'

        if manager.wsd:
            word_labels_by_idx = OrderedDict()
            word_keys_by_idx = OrderedDict()
            for word in self:
                sense_data = word.word_sense_data
                if sense_data.item_id in word_keys_by_idx:
                    assert word_labels_by_idx[sense_data.item_id] == sense_data.gold_sense_idx, \
                        'Same ID words must share gold sense'
                    assert word_keys_by_idx[sense_data.item_id] == sense_data.key_id, 'Same ID words must share key id'
                else:
                    word_labels_by_idx[sense_data.item_id] = sense_data.gold_sense_idx
                    word_keys_by_idx[sense_data.item_id] = sense_data.key_id

            self.word_sense_labels = torch.tensor(list(word_labels_by_idx.values()), dtype=torch.long)
            self.word_sense_keys = torch.tensor(list(word_keys_by_idx.values()), dtype=torch.long)

            assert len(set(x.item() for x in self.token_word_ids) - {SENSE_MASK_ID}) == self.word_sense_keys.shape[0]
            assert (self.word_sense_labels != SENSE_MASK_ID).sum().item() == self.wsd_count, \
                'Must have sense per labeled word'
            assert (self.word_sense_keys != SENSE_MASK_ID).sum().item() == self.wsd_count, \
                'Must have key per labeled word'

        if manager.mwe:
            token_mwe_ids = [w.mwe_id_tensor for w in words]
            token_mwe_ids = [torch.tensor([SENSE_MASK_ID])] + token_mwe_ids + [torch.tensor([SENSE_MASK_ID])]
            self.token_mwe_ids = torch.cat(token_mwe_ids)

            assert self.input_ids.shape == self.token_mwe_ids.shape, 'Must be an MWE ID for every input ID'

            mwe_sense_labels = [set() for _ in range(self.mwe_count)]
            mwe_sense_keys = [set() for _ in range(self.mwe_count)]

            for word in self:
                sense_data = word.mwe_sense_data
                if sense_data is not None and sense_data.labeled:
                    mwe_sense_labels[sense_data.item_id].add(sense_data.gold_sense_idx)
                    mwe_sense_keys[sense_data.item_id].add(sense_data.key_id)

            assert all(len(s) == 1 for s in mwe_sense_keys), 'Must be exactly one key per MWE'
            assert all(len(s) == 1 for s in mwe_sense_labels), 'Must be exactly one label per MWE'

            self.mwe_sense_labels = torch.tensor([next(iter(s)) for s in mwe_sense_labels], dtype=torch.long)
            self.mwe_sense_keys = torch.tensor([next(iter(s)) for s in mwe_sense_keys], dtype=torch.long)

            assert len(set(x.item() for x in self.token_mwe_ids) - {SENSE_MASK_ID}) == self.mwe_sense_keys.shape[0]
            assert (self.mwe_sense_labels != SENSE_MASK_ID).sum().item() == self.mwe_count, 'Must have sense per MWE'
            assert (self.mwe_sense_keys != SENSE_MASK_ID).sum().item() == self.mwe_count, 'Must have key per MWE'

    def to_json(self) -> Dict:
        return {
            'text': self.original_text,
            'words': [w.to_json() for w in self],
            'meta': self._metadata if self._metadata else None
        }

    @property
    def metadata(self) -> Dict:
        if self._metadata is None:
            self._metadata = {}

        return self._metadata

    @staticmethod
    def from_json(json: Dict, manager: Optional[TrainingDefinitionManager]) -> TrainingSentence:
        return TrainingSentence([TrainingWord.from_json(word, manager) for word in json['words']],
                                json['text'], manager, metadata=json.get('meta', None))

    @property
    def wsd_labeled_word_count(self) -> int:
        return sum(1 for word in self if word.word_sense_data.labeled)

    @property
    def mwe_labeled_word_count(self) -> int:
        return sum(1 for word in self if word.mwe_sense_data.labeled)

    @property
    def mwe_count(self) -> int:
        return len(set(w.mwe_sense_data.item_id for w in self if w.mwe_sense_data is not None) - {SENSE_MASK_ID})

    def get_mwe_groups(self, added_is: Optional[bool] = None) -> List[Tuple[SenseData, List[TrainingWord]]]:
        mwe_ids = set(w.mwe_sense_data.item_id for w in self if w.mwe_sense_data is not None) - {SENSE_MASK_ID}
        mwe_word_groups = [
            [w for w in self if w.mwe_sense_data is not None and w.mwe_sense_data.item_id == mwe_id]
            for mwe_id in mwe_ids
        ]

        return [
            (word_group[0].mwe_sense_data, word_group) for word_group in mwe_word_groups
            if added_is is None or added_is == word_group[0].mwe_sense_data.metadata.get('added', False)
        ]

    @property
    def wsd_count(self) -> int:
        return len(set(w.word_sense_data.item_id for w in self if w.word_sense_data.labeled) - {SENSE_MASK_ID})


class TrainingDefinitionManager:
    def _no_set(self, name: str, _: Any):
        raise RuntimeError(f"Cannot change variable {name} on TrainingDataManager")

    def __init__(self, context_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 definition_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 allow_single_def: bool, def_language: str, mwe_training: bool = False,
                 wsd_training: bool = True):
        assert wsd_training or mwe_training, 'Must be traiing for either WSD or MWE or both'

        self.context_tokenizer = context_tokenizer
        self.definition_tokenizer = definition_tokenizer
        self.allow_single_def = allow_single_def
        self.def_language = def_language
        self.mwe = mwe_training
        self.wsd = wsd_training

        self._definition_batches: Dict[int, BatchEncoding] = {}
        self._key_vocabulary = {}
        self._next_key_id = 0

        self.__setattr__ = self._no_set

    def register_word_key(self, key: str, definitions: List) -> int:
        if key not in self._key_vocabulary:
            self._key_vocabulary[key] = self._next_key_id
            self._next_key_id += 1
            self._definition_batches[self._key_vocabulary[key]] = self.definition_tokenizer(definitions,
                                                                                            return_tensors='pt',
                                                                                            padding=True)

        return self._key_vocabulary[key]

    def get_definition_batch(self, key_id: int) -> BatchEncoding:
        output = self._definition_batches[key_id]
        assert output['input_ids'].device.type == 'cpu', 'Definition batches should initially be on CPU'
        return output


@dataclass
class DataSummaryStats:
    wsd_ignored: int = 0
    wsd_active: int = 0
    mwe_ignored: int = 0
    mwe_active: int = 0
    mwe_active_negative: int = 0
    active_sentences: int = 0
    mwe_added_but_not_annotated: int = 0
    ignored_sentences: Optional[int] = None
    batch_count: Optional[int] = None

    def to_dict(self) -> Dict:
        output = dataclasses.asdict(self)

        if self.mwe_active > 0:
            output['mwe_negative_percent'] = to_percent(self.mwe_active_negative / self.mwe_active)
            output['mwe_percent_of_total'] = to_percent(self.mwe_active / (self.mwe_active + self.wsd_active))
        if self.mwe_ignored > 0:
            output['mwe_ignored_percent'] = to_percent(self.mwe_ignored / (self.mwe_active + self.mwe_ignored))
        if self.wsd_ignored > 0:
            output['wsd_ignored_percent'] = to_percent(self.wsd_ignored / (self.wsd_active + self.wsd_ignored))
        if self.ignored_sentences is not None and self.ignored_sentences > 0:
            output['sent_ignored_percent'] = to_percent(self.ignored_sentences /
                                                        (self.active_sentences + self.ignored_sentences))

        return {k: v for k, v in output.items() if v is not None}


def to_percent(f: float) -> str:
    return "{0:.2%}".format(f)


def compute_summary_stats(sentences: Iterable[TrainingSentence], ignored_sentences: Optional[int] = None,
                          batch_count: Optional[int] = None, no_tqdm: bool = False) -> DataSummaryStats:
    summary = DataSummaryStats(ignored_sentences=ignored_sentences, batch_count=batch_count)
    for sentence in tqdm(sentences, 'computing summary stats', disable=no_tqdm):
        summary.active_sentences += 1
        for word in sentence:
            if word.word_sense_data is not None and word.word_sense_data.gold_sense is not None:
                if word.word_sense_data.ignored:
                    summary.wsd_ignored += 1
                else:
                    summary.wsd_active += 1

        for mwe_sense, _ in sentence.get_mwe_groups():
            if mwe_sense.added_but_not_annotated:
                summary.mwe_added_but_not_annotated += 1
                continue

            if mwe_sense.ignored:
                summary.mwe_ignored += 1
            else:
                summary.mwe_active += 1
                if mwe_sense.is_mwe_negative:
                    summary.mwe_active_negative += 1

    return summary


class DefinitionMatchingDataset(IterableDataset):

    def __init__(self, datasets: List[Path], manager: TrainingDefinitionManager, batch_size: int,
                 disable_tqdm: bool = False, suffix: Optional[str] = None, upsample: Optional[List[int]] = None,
                 **kwargs):
        suffix = '' if suffix is None else f'{suffix}.'
        data_paths = [x / f'{x.name.lower()}.{suffix}jsonl' for x in datasets]

        self.dataset_paths = datasets
        if upsample is not None:
            print(f'Found upsample rates {upsample} for datasets {datasets}')
            assert(len(upsample) == len(datasets)), 'Must be one upsample value for each dataset'
        else:
            upsample = repeat(1)

        assert all(x.exists() for x in data_paths), f'One of {data_paths} is missing'
        self.rand_state = random.Random(42)
        self.batch_size = batch_size
        self.manager = manager

        self.sentences = []
        self.ignored_sentences = []

        for dataset_path, upsample_rate in zip(data_paths, upsample):
            for _ in range(upsample_rate):
                for sentence_data in read_training_sentences(dataset_path, manager, disable_tqdm=disable_tqdm):
                    if (manager.wsd and sentence_data.wsd_count != 0) or (manager.mwe and sentence_data.mwe_count != 0):
                        self.sentences.append(sentence_data)
                    else:
                        self.ignored_sentences.append(sentence_data)

        self._len = sum(1 for _ in self)  # compute length in advance because it requires running through the dataset
        self._summary_stats = compute_summary_stats(self.sentences, len(self.ignored_sentences), self._len)

    @property
    def summary_stats(self) -> Dict:
        return self._summary_stats.to_dict()

    def __len__(self):
        return self._len

    def __iter__(self) -> Iterator[List[TrainingSentence]]:
        if len(self.sentences) == 0:
            return

        # make sort order random inside length, then make batches with similar lengths
        sorted_sents = sorted(self.sentences, key=lambda s: s.input_ids.shape[0] + self.rand_state.random())
        all_batches = [sorted_sents[i:i + self.batch_size] for i in range(0, len(sorted_sents), self.batch_size)]

        # longest batch is first, otherwise order of batches is random
        first_batch = all_batches[-1]
        remaining_batches = all_batches[:-1]
        self.rand_state.shuffle(remaining_batches)
        yield first_batch
        for batch in remaining_batches:
            yield batch

    def collate_into_batch(self, sentences: List[TrainingSentence]) -> Dict[str, torch.Tensor]:
        """
        S = [sentence count]
        T = [BPE token count]
        W = [word count]

        These three tensors are S X T
         - sentence_ids: Input ids for context(s)
         - sentence_attention_mask: Attention mask for contexts
         - word_ids: Ids mapping words to wordpieces

         These two tensors are S X W
         - word_sense_keys: Keys to look up possible word senses, at indices of target words
         - word_sense_labels: Index of correct senses for each word"""

        # these three tensors are B
        sentence_ids = pad_sequence([sentence.input_ids for sentence in sentences], True,
                                    self.manager.context_tokenizer.pad_token_id)
        sentence_attention_mask = (sentence_ids != self.manager.context_tokenizer.pad_token_id).int()
        output = {
            'sentence_ids': sentence_ids,
            'sentence_attention_mask': sentence_attention_mask,
        }

        if self.manager.wsd:
            output['word_ids'] = pad_word_sequence([sentence.token_word_ids for sentence in sentences])
            output['word_sense_keys'] = pad_word_sequence([sentence.word_sense_keys for sentence in sentences])
            output['word_sense_labels'] = pad_word_sequence([sentence.word_sense_labels for sentence in sentences])

        if self.manager.mwe:
            output['mwe_ids'] = pad_word_sequence([sentence.token_mwe_ids for sentence in sentences])
            output['mwe_sense_keys'] = pad_word_sequence([sentence.mwe_sense_keys for sentence in sentences])
            output['mwe_sense_labels'] = pad_word_sequence([sentence.mwe_sense_labels for sentence in sentences])

        return output

    def words_for_batch(self, sentences: List[TrainingSentence],
                        batch: Dict[str, torch.Tensor]) -> List[List[TrainingWord]]:
        assert self.manager.allow_single_def and len(self.dataset_paths) == 1, \
            'Can only reconstruct words for a single dataset that allows single definition words'
        word_matrix = [list(sent) for sent in sentences]
        output = []
        for word_list, word_sense_keys in zip(word_matrix, batch['word_sense_keys']):
            word_item_ids = (word_sense_keys != SENSE_MASK_ID).nonzero().flatten()
            # Multiple words may sometimes have the same item IDS (MWEs in the WSD data)
            # take the first word only, but we have one label per item ID and grouped words should share WSD IDs
            # so this shouldn't be an issue
            target_words = [
                next(w for w in word_list if w.word_sense_data.item_id == item_id)
                for item_id in word_item_ids
            ]
            assert all(w.word_sense_data.labeled for w in target_words), 'Returned words should be labeled'
            output.append(target_words)

        return output

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError


def pad_word_sequence(tensors: List[torch.Tensor]) -> torch.Tensor:
    return pad_sequence(tensors, True, SENSE_MASK_ID)


class FixedCountBatch(List[TrainingSentence]):
    def __init__(self, sentences: List[TrainingSentence], item_offset: int, repeated_sentence: List[bool]):
        super(FixedCountBatch, self).__init__(sentences)
        self.item_offset = item_offset
        self.repeated_sentence = repeated_sentence
        assert len(self) == len(repeated_sentence)


class DefinitionMatchingFixedCountBatchDataset(DefinitionMatchingDataset):
    WORD_COUNT = lambda _, x: x.wsd_count
    MWE_COUNT = lambda _, x: x.mwe_count

    def __init__(self, *args, **kwargs):
        count_wsd = kwargs.get('count_wsd', True)
        if count_wsd:
            self._count_function = self.WORD_COUNT
            self._primary_prefix = 'word'
            self._secondary_prefix = 'mwe'
        else:
            self._count_function = self.MWE_COUNT
            self._primary_prefix = 'mwe'
            self._secondary_prefix = 'word'

        # this has to come last since we need the count function to iterate, to compute length
        super(DefinitionMatchingFixedCountBatchDataset, self).__init__(*args, **kwargs)
        if count_wsd:
            assert self.manager.wsd, 'Cannot count WSD if not training with WSD'
        else:
            assert self.manager.mwe, 'Cannot count MWE if not training with MWE'

    def __iter__(self) -> Iterator[FixedCountBatch]:
        if len(self.sentences) == 0:
            return

        # make sort order random inside length, then make batches with similar lengths
        sorted_sents = sorted(self.sentences, key=lambda s: s.input_ids.shape[0] + self.rand_state.random())
        seen_sents = set()

        all_batches = []
        current_batch = []
        current_sents_repeated = []
        item_count = 0
        rollover_offset = 0

        for sent in tqdm(sorted_sents, desc='precomputing iterations', leave=False):
            labeled_item_count = self._count_function(sent)
            if item_count + labeled_item_count >= self.batch_size:
                # if there are enough items in this sentence to finish at least the current batch, maybe more
                items_required = self.batch_size - item_count
                offset = 0

                while offset < labeled_item_count:
                    words_added = min(labeled_item_count - offset, items_required)

                    if words_added == items_required:  # if we completed a batch
                        current_batch.append(sent)
                        current_sents_repeated.append(sent in seen_sents)
                        seen_sents.add(sent)

                        if current_batch[0] == sent:
                            all_batches.append(FixedCountBatch(current_batch, offset, current_sents_repeated))
                        else:
                            all_batches.append(FixedCountBatch(current_batch, rollover_offset, current_sents_repeated))
                            rollover_offset = 0

                        current_batch = []
                        current_sents_repeated = []
                        items_required = self.batch_size
                        offset += words_added
                        item_count = 0
                    else:
                        assert len(current_batch) == 0, 'Should only happen with empty batch'
                        current_batch = [sent]
                        current_sents_repeated = [sent in seen_sents]
                        seen_sents.add(sent)
                        item_count = words_added
                        rollover_offset = offset

                        offset += words_added
                        assert offset >= labeled_item_count, 'Should only happen on final iteration'

            else:
                # words left in this sentence won't finish the batch
                current_batch.append(sent)
                current_sents_repeated.append(sent in seen_sents)
                seen_sents.add(sent)
                item_count += labeled_item_count

        if len(current_batch) > 0:
            all_batches.append(FixedCountBatch(current_batch, rollover_offset, current_sents_repeated))

        # longest batch is first, otherwise order of batches is random
        first_batch = all_batches[-1]
        remaining_batches = all_batches[:-1]
        self.rand_state.shuffle(remaining_batches)
        yield first_batch
        for batch in remaining_batches:
            yield batch

    def collate_into_batch(self, batch: FixedCountBatch) -> Dict[str, torch.Tensor]:
        """Masks out primary item senses so that we have exactly (or for the first batch, at most) self.batch_size
        senses in the batch. Note that we cannot simultaneously guarantee the number of normal WSD words and MWEs
        in the batch at the same time, so for the case where we are training with MWEs as a separate input, the
        number of MWEs in the batch is unfortunately random (or vice versa if MWEs are primary).
        We do make sure that we don't show the model the same MWE twice, though."""
        base_batch = super().collate_into_batch(batch)

        # get locations of labeled primary items, then select the items for this batch from those
        labeled_primary_locations = (base_batch[f'{self._primary_prefix}_sense_keys'] != SENSE_MASK_ID).nonzero()
        batch_primary_locations = labeled_primary_locations[batch.item_offset:batch.item_offset + self.batch_size]

        # create a mask on all indices except the selected primary items
        locations_to_mask = torch.full(base_batch[f'{self._primary_prefix}_sense_keys'].shape, True)
        locations_to_mask[tuple(batch_primary_locations.transpose(0, 1))] = False

        # mask out all indices except target primary items (most of these will be WORD_MASK_ID to begin with)
        base_batch[f'{self._primary_prefix}_sense_keys'][locations_to_mask] = SENSE_MASK_ID
        base_batch[f'{self._primary_prefix}_sense_labels'][locations_to_mask] = SENSE_MASK_ID

        if self.manager.mwe and self.manager.wsd:
            ignore_sentence_mask = torch.tensor(batch.repeated_sentence)
            # this lets us mask any repeated sentences so we don't show the model the same secondary multiple times
            base_batch[f'{self._secondary_prefix}_ids'][ignore_sentence_mask] = SENSE_MASK_ID
            base_batch[f'{self._secondary_prefix}_sense_keys'][ignore_sentence_mask] = SENSE_MASK_ID
            base_batch[f'{self._secondary_prefix}_sense_labels'][ignore_sentence_mask] = SENSE_MASK_ID

        # count should generally be equal to batch size, but the first batch may have fewer words
        primary_key_count = (base_batch[f'{self._primary_prefix}_sense_keys'] != SENSE_MASK_ID).sum()
        primary_label_count = (base_batch[f'{self._primary_prefix}_sense_labels'] != SENSE_MASK_ID).sum()
        assert 0 < primary_key_count <= self.batch_size
        assert 0 < primary_label_count <= self.batch_size

        return base_batch

    def words_for_batch(self, batch: FixedCountBatch,
                        collated_batch: Dict[str, torch.Tensor]) -> List[List[TrainingWord]]:
        return super().words_for_batch(batch, collated_batch)

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError


class DefinitionMatchingLoader(pl.LightningDataModule):

    def __init__(self, datasets: Dict[str, List[Path]],
                 context_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], batch_size: int,
                 dataset_type: Type[DefinitionMatchingDataset],
                 definition_tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
                 allow_single_def: bool = False, definition_language: str = 'eng', include_mwe: bool = False,
                 include_wsd: bool = True, train_data_suffix: Optional[str] = None,
                 upsample_train: Optional[List[int]] = None):
        # note that allow_single_def only applies to WSD, not MWE senses (single def is always okay for them)
        super().__init__()

        if definition_tokenizer is None:
            definition_tokenizer = context_tokenizer

        self.manager = TrainingDefinitionManager(context_tokenizer, definition_tokenizer,
                                                 allow_single_def, definition_language,
                                                 mwe_training=include_mwe, wsd_training=include_wsd)

        self.train_data = dataset_type(datasets['train'], self.manager, batch_size, suffix=train_data_suffix,
                                       count_wsd=include_wsd, upsample=upsample_train)
        self.dev_data = dataset_type(datasets['dev'], self.manager, batch_size, count_wsd=include_wsd)
        self.test_data = dataset_type(datasets['test'], self.manager, batch_size, count_wsd=include_wsd)

        print('SUMMARY STATS FOR DATASETS')
        for dataset, name in [
            (self.train_data, 'train'),
            (self.dev_data, 'dev'),
            (self.test_data, 'test')
        ]:
            print('Summary stats for', name)
            print('---------------------------')
            pp(dataset.summary_stats)
            print('---------------------------')

    def get_summary_stats_dict(self):
        return {
            'train_dataset': self.train_data.summary_stats,
            'dev_dataset': self.dev_data.summary_stats,
            'test_dataset': self.test_data.summary_stats
        }

    @property
    def train_sense_inventory(self) -> Dict:
        return self.train_data.def_batches_by_id

    @property
    def test_sense_inventory(self) -> Dict:
        return self.test_data.def_batches_by_id

    @property
    def val_sense_inventory(self) -> Dict:
        return self.dev_data.def_batches_by_id

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, batch_size=None, collate_fn=self.train_data.collate_into_batch)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, batch_size=None, collate_fn=self.test_data.collate_into_batch)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dev_data, batch_size=None, collate_fn=self.dev_data.collate_into_batch)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError


class Detokenizer:
    _detokenizer = TreebankWordDetokenizer()
    _quote_regex = re.compile(r"([\"'])(?:(?=(\\?))\2.)*?\1")

    @classmethod
    def detokenize(cls, words: List[TrainingWord]) -> str:
        # reconstruct original text (not perfect, but the best we can do)
        detokenized = cls._detokenizer.detokenize([x.form for x in words])
        char_list = list(detokenized)
        quote_locations = cls._quote_regex.finditer(detokenized)

        # fixes the bug where we get misplaced quotes like 'He said "I like dogs "but I know he is lying'
        for match in quote_locations:
            end = match.span()[1] - 1
            if end + 1 < len(char_list) and not char_list[end + 1].isspace() and char_list[end - 1].isspace():
                space_char = char_list[end - 1]
                char_list[end - 1] = char_list[end]  # move quote back one char
                char_list[end] = space_char  # move space up one char

        return ''.join(char_list)
