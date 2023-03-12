from __future__ import annotations

import datetime
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, NamedTuple, OrderedDict, Optional, Union, Dict, Iterable

import nltk
from nltk.corpus import wordnet as wn, WordNetCorpusReader
from tqdm import tqdm
from spacy import parts_of_speech

from resolve.common.data import WordnetDefinitionLookup

wordnet_pos_map = {
    'v': parts_of_speech.IDS['VERB'],
    'a': parts_of_speech.IDS['ADJ'],
    'n': parts_of_speech.IDS['NOUN'],
    'r': parts_of_speech.IDS['ADV'],
    's': parts_of_speech.IDS['ADJ']  # "adjective satellite"
}


class OrderedCounter(Counter, OrderedDict):
    pass


# needs to be a named tuple and not dataclass so it can be unpacked by sqlite
class MWEData(NamedTuple):
    lemma: str
    lemma_cased: str
    pos: str
    word_count: str
    uninflected: Optional[int] = None
    metadata: Optional[str] = None

    def get_definitions(self, lang: str, ignore_candidates: bool = False) -> Optional[List[str]]:
        return WordnetDefinitionLookup.get_definitions(self.lemma, self.pos, lang, ignore_candidates)

    @property
    def lemma_counter(self) -> Counter:
        return Counter(self.lemma_iter)

    @property
    def lemma_indices(self) -> Dict[str, List[int]]:
        output = defaultdict(list)
        for idx, part in enumerate(self.lemma_iter):
            output[part].append(idx)

        return output

    @property
    def lemma_iter(self) -> Iterable[str]:
        for part in self.lemma.split('_'):
            yield part

    def key(self):
        return self.lemma, self.pos


def load_old_wordnet(wordnet_name: str) -> WordNetCorpusReader:
    wn_loc = nltk.data.find(wordnet_name).path
    dict_loc = wn_loc + '/dict'
    try:
        return WordNetCorpusReader(nltk.data.find(dict_loc), None)
    except OSError:
        wn_path = Path(wn_loc)
        wn_subdir_path = (wn_path / 'wordnet')
        wn_subdir_path.mkdir(exist_ok=True)
        shutil.copy(wn_path / 'dict' / 'index.sense', wn_subdir_path / 'index.sense')
        return WordNetCorpusReader(nltk.data.find(dict_loc), None)


class MWEIndex:

    @classmethod
    def _wordnet_mwe_entries(cls, wordnet_name: Optional[str]) -> List[MWEData]:
        mwe_data_list = []

        if wordnet_name is not None:
            wordnet = load_old_wordnet(wordnet_name)
        else:
            wordnet = wn

        for synset in tqdm(wordnet.all_synsets(), 'Reading from wordnet'):
            pos = synset.pos()
            for lemma in synset.lemmas():
                if '_' in lemma.name():
                    mwe_data_list.append(MWEData(lemma.name().lower(), lemma.name(), pos, str(len(lemma.name().split('_')))))

        return mwe_data_list

    @classmethod
    def _datafile_mwe_entries(cls, filepath: Path) -> List[MWEData]:
        pos_conversion_dict = {
            'N': wn.NOUN,
            'J': wn.ADJ,
            'V': wn.VERB,
            'O': 'o',  # articles, not in new wordnet (?)
            'R': wn.ADV
        }

        with filepath.open('r') as data_file:
            data = list(data_file.readlines())

        mwe_entry_data = []
        for line_data in data:
            if line_data.startswith('//'):  # a comment line
                continue

            tokens = line_data.split()
            assert len(tokens) % 2 == 0
            entry_data_for_line = []
            for i in range(0, len(tokens), 2):
                entry, counts = tokens[i:i+2]

                pieces = entry.split('_')
                if i == 0:
                    pos = pos_conversion_dict[pieces[-1]]  # POS only included in first entry
                    lemma = '_'.join(pieces[:-1])
                    uninflected = None
                else:
                    lemma = '_'.join(pieces)
                    uninflected = entry_data_for_line[0]

                assert lemma
                metadata = json.dumps({'counts': counts})

                entry_data_for_line.append(MWEData(lemma.lower(), lemma, pos,
                                                   str(len(lemma.split('_'))), uninflected, metadata))
            mwe_entry_data.extend(entry_data_for_line)

        return mwe_entry_data

    @classmethod
    def get_or_build_index(cls, index_path: Path, input_data: Optional[Union[str, Path]] = None,
                           in_memory: bool = False):
        if not index_path.exists():
            cls.build_fresh_index(index_path, input_data)

        return MWEIndex(index_path, in_memory)

    @classmethod
    def build_fresh_index(cls, index_path: Path, input_data: Optional[Union[str, Path]]):
        assert not index_path.exists()
        con = sqlite3.connect(index_path)
        cur = con.cursor()

        cur.execute('CREATE TABLE mwes (lemma TEXT, lemma_cased TEXT, pos TEXT, word_count INTEGER, '
                    'uninflected INTEGER, metadata TEXT,'
                    ' FOREIGN KEY(uninflected) REFERENCES mwes(rowid) UNIQUE(lemma, pos))')
        cur.execute('CREATE TABLE mwe_words (lemma TEXT UNIQUE)')
        cur.execute('CREATE TABLE mwe_word_assoc (mwe_id INTEGER REFERENCES mwes, word_id INTEGER REFERENCES words)')
        cur.execute('CREATE TABLE metadata (key TEXT, val TEXT)')

        if isinstance(input_data, Path):
            print(f'Building from file data source {input_data}')
            entries = cls._datafile_mwe_entries(Path(input_data))
        else:
            print(f'Building from Wordnet data source {input_data}')
            entries = cls._wordnet_mwe_entries(input_data)

        seen_words = {}
        rowids = {}
        dupes = []
        input_entry_count = 0
        for entry in tqdm(entries, f'Writing MWE entries to {index_path}'):
            input_entry_count += 1
            try:
                if entry.key() in rowids:
                    dupes.append(entry)
                    continue

                if entry.uninflected is not None:
                    entry = entry._replace(uninflected=rowids[entry.uninflected.key()][0])

                words = entry.lemma.split('_')
                assert len(words) > 1
                cur.execute('INSERT INTO mwes VALUES (?, ?, ?, ?, ?, ?)', entry)
                mwe_id = cur.lastrowid
                rowids[entry.key()] = (mwe_id, entry)
                for word in words:
                    if word not in seen_words:
                        cur.execute('INSERT INTO mwe_words VALUES (?)', (word,))
                        word_id = cur.lastrowid
                        seen_words[word] = word_id
                    else:
                        word_id = seen_words[word]

                    cur.execute('INSERT INTO mwe_word_assoc VALUES (?, ?)', (mwe_id, word_id))
            except sqlite3.Error as err:
                print(f'Failed to insert {entry} due to {err}')

        for key, val in {
            'input_entries': input_entry_count,
            'duplicate_entries': len(dupes),
            'created_date': datetime.date.today(),
            'created_path': index_path
        }.items():
            cur.execute('INSERT INTO metadata VALUES (?, ?)', (key, str(val)))

        con.commit()
        cur.close()
        con.close()
        for dupe in dupes:
            print(f'{dupe} skipped as duplicate of {rowids[dupe.key()][1]}')

    def connect(self, in_memory: bool):
        if in_memory:
            source = sqlite3.connect(self.index_path)
            con = sqlite3.connect(':memory:')
            source.backup(con)
            source.close()
        else:
            con = sqlite3.connect(self.index_path)

        self.con = con
        self.cur = self.con.cursor()

    def disconnect(self):
        self.cur.close()
        self.cur = None
        self.con.close()
        self.con = None

    def __init__(self, index_path: Path, in_memory: bool = False):
        if not index_path.exists():
            raise FileNotFoundError(f'Index is missing from {index_path}')

        self.index_path = index_path
        self.con = None
        self.cur = None
        self.connect(in_memory)

    def metadata(self) -> Dict[str, str]:
        return {
            x[0]: x[1] for x in self.cur.execute('SELECT * FROM metadata')
        }

    def possible_mwes_for_lemmas(self, lemmas: List[str]) -> List[MWEData]:
        query = f"""
        SELECT mwes.* FROM mwe_words 
        JOIN mwe_word_assoc ON mwe_word_assoc.word_id = mwe_words.rowid
        JOIN mwes ON mwe_word_assoc.mwe_id == mwes.rowid
        WHERE mwe_words.lemma IN ({", ".join("?" * len(lemmas))})
        GROUP BY mwes.rowid
        HAVING COUNT(mwe_words.rowid) == word_count
        """
        
        word_query = self.cur.execute(query, lemmas)

        return [
            MWEData(*row) for row in word_query.fetchall()
        ]






