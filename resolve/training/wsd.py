from __future__ import annotations
from io import BytesIO
from pathlib import Path, WindowsPath

from lxml.etree import parse, ElementTree, Element
from tqdm import tqdm
from jsonlines import open as json_open


from resolve.training.data import TrainingWord, SENSE_MASK_ID, TrainingSentence, SenseData, Detokenizer
from resolve.common.data import UPOS_TO_WN_POS_MAP


class WSDWord(TrainingWord):

    def __init__(self, form: str, element: Element, idx: int):
        assert element.tag == 'wf' or element.tag == 'instance'
        attributes = element.attrib

        word_id = attributes.get('id', None)
        gold_sense = attributes['gold_sense'] if word_id is not None else None

        pos = UPOS_TO_WN_POS_MAP.get(attributes['pos'], attributes['pos'])
        word_lemma = attributes['lemma']
        word_sense_data = SenseData(word_lemma, pos, idx, gold_sense, None, False)

        if word_id is not None:
            word_sense_data.metadata['wsd_id'] = word_id

        if '_' in word_lemma:
            mwe_sense_data = SenseData(attributes['lemma'], pos, SENSE_MASK_ID, gold_sense, None, True)
        else:
            mwe_sense_data = None

        super(WSDWord, self).__init__(form, word_sense_data, mwe_sense_data, None)


class WSDSentence(TrainingSentence):

    def __init__(self, element: Element):
        assert element.tag == 'sentence'

        words = []
        for idx, word_elem in enumerate(element):
            # this means splits will produce multiple words with the same id, which reflects the original data
            words.extend(WSDWord(word_form, word_elem, idx) for word_form in word_elem.text.split())

        # any MWEs at this stage were marked as the same word in the original data, so we use the word IDX
        mwe_groups = [
            [word for word in words if word.word_sense_data.item_id == word_idx]
            for word_idx in {word.word_sense_data.item_id for word in words if word.mwe_sense_data is not None}
        ]

        # assign initial MWE item IDS
        for idx, mwe_words in enumerate(mwe_groups):
            for word in mwe_words:
                word.mwe_sense_data.item_id = idx

        original_text = Detokenizer.detokenize(words)

        super(WSDSentence, self).__init__(words, original_text, None)


def wrapped_xml_file(filepath: Path) -> ElementTree:
    if isinstance(filepath, WindowsPath):
        filepath = str(filepath).replace('\\', '/')
    wrapper_string = f"""<!DOCTYPE wrapper [
    <!ENTITY e SYSTEM "{filepath}">
    ]>
    <wrapper>&e;</wrapper>"""
    wrapper_bytes = BytesIO(bytes(wrapper_string, 'utf-8'))
    return parse(wrapper_bytes)


def preprocess_xml(dataset_path: Path):

    dataset_name = dataset_path.name.lower()

    data_path = dataset_path / '{}.data.xml'.format(dataset_name)
    gold_path = dataset_path / '{}.gold.key.txt'.format(dataset_name)
    assert data_path.exists(), f'{data_path} is missing'
    assert gold_path.exists(), f'{gold_path} is missing'

    gold_senses = {}

    with gold_path.open() as gold_data_file:
        for line in tqdm(gold_data_file, gold_path.name):
            entries = line.strip().split()
            word_id = entries[0]
            # some entries are labeled with multiple senses, but we take the first sense to be the gold label
            sense = entries[1]
            gold_senses[word_id] = sense

    tree = wrapped_xml_file(data_path)
    input_name = data_path.name
    output_path = data_path.parent / ('.'.join(input_name.split('.')[:-2]) + '.jsonl')
    with json_open(output_path.absolute(), 'w') as outfile:
        for corpus_element in tree.getroot():
            assert corpus_element.tag == 'corpus'
            for text in tqdm(corpus_element, f'{input_name} ({corpus_element.attrib["source"]})'):
                for sentence in tqdm(text, text.attrib['id'], leave=False):
                    for word_elem in sentence:
                        if 'id' in word_elem.attrib:
                            word_elem.attrib['gold_sense'] = gold_senses[word_elem.attrib['id']]

                    sentence_data = WSDSentence(sentence)
                    outfile.write(sentence_data.to_json())

    print('Wrote to', output_path)


