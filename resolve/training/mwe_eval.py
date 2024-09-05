import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Dict

from tqdm import tqdm

from resolve.mwe.detect import KulkarniPropNounDetector, DimSumSequentialNounDetector
from resolve.mwe.filter import OrderedOnly, MaxGappiness, SpecificPOSOnly, ModelOutputIsMWE, FilterEvalData
from resolve.mwe.index import MWEIndex
from resolve.mwe.pipeline import MWEPipeline, PipelineOutput
from resolve.mwe.resolve import ModelScoreResolver, KulkarniResolverWrapper, LessGappyLongerResolver
from resolve.training import DATA_DIR
from resolve.training.data import read_training_sentences, TrainingSentence
from sklearn.metrics import precision_recall_fscore_support, classification_report

from resolve.model.pl_module import ContextDictionaryBiEncoder


@dataclass(frozen=True)
class EvalLabels:
    predicted: List[int]
    gold: List[int]
    eval_data: Optional[Dict[str, List[FilterEvalData]]]
    pipeline_results: PipelineOutput
    false_negatives: List
    false_positives: List
    partial_matches: List


@dataclass(frozen=True)
class EvalStats:
    precision: float
    recall: float
    f1: float


class MWEEvaluator:

    def __init__(self, pipeline: MWEPipeline, data: List[TrainingSentence], **kwargs):
        self.pipeline = pipeline
        self.pipeline.compute_eval_stats = True
        self.data = data
        self.printing = kwargs.get('print', False)
        self.examples: Path = kwargs.get('examples', None)
        if self.printing:
            self.printing_filename = f'eval_out_{int(time.time())}.txt'
            self.file = open(self.printing_filename, 'w')
            self.file.write('WORD\tPOS\tHYP_LEM\tGOLD_LEM\tHYP\tGOLD\n\n')

        if self.examples:
            self.example_file = self.examples.open('w')

    @staticmethod
    def _compute_aligned_eval_output(gold_labels: List[int], predicted_labels: List[int]) -> EvalStats:
        labels = list((set(gold_labels) | set(predicted_labels)) - {0})
        precision, recall, f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels,
                                                                   average='micro', labels=labels)

        return EvalStats(precision, recall, f1)

    @staticmethod
    def _compute_binary_eval_output(gold_labels: List[int], predicted_labels: List[int]) -> EvalStats:
        """This is less meaningful, but it's what Kulkarni reports - just binary MWE or not label"""
        binarized_gold_labels = [0 if label == 0 else 1 for label in gold_labels]
        binarized_predicted_labels = [0 if label == 0 else 1 for label in predicted_labels]

        precision, recall, f1, _ = precision_recall_fscore_support(binarized_gold_labels, binarized_predicted_labels,
                                                                   average='binary')

        return EvalStats(precision, recall, f1)

    def labels_for_sentence(self, sentence: TrainingSentence) -> EvalLabels:
        mismatch_label_iter = iter(reversed(range(-10000, -1)))

        false_positives = []
        partial_mismatch = []

        sent_results = self.pipeline(sentence)
        predicted_labels = [0] * len(sentence)
        touched_ids = set()
        for found_mwe in sent_results:
            word_mwe_ids = [w.mwe_sense_data.item_id for w in found_mwe.words if w.mwe_sense_data]
            most_common_id = Counter(word_mwe_ids).most_common(1)

            # in the case of no overlap with any MWE, choose an ID <-1 which is guaranteed to be wrong
            # item IDs start from 0, but 0 is the negative label
            mwe_id = most_common_id[0][0] + 1 if len(most_common_id) > 0 else next(mismatch_label_iter)
            if mwe_id < 0:
                false_positives.append((found_mwe.words, found_mwe.mwe_data.lemma))
            else:
                touched_ids.add(mwe_id)
                if len(set(word_mwe_ids)) > 1:
                    partial_mismatch.append((found_mwe.words, found_mwe.mwe_data.lemma))

            for word in found_mwe.words:
                predicted_labels[word.idx] = mwe_id
                word.mwe_candidate = found_mwe

        # item IDs start from 0, but 0 is the negative label
        gold_labels = [w.mwe_sense_data.item_id + 1 if w.mwe_sense_data else 0 for w in sentence]

        missed_ids = set(gold_labels) - touched_ids - {0}
        false_negatives = [
            (words, words[0].mwe_sense_data.lemma) for words in
            ([w for w in sentence if w.mwe_sense_data and w.mwe_sense_data.item_id + 1 == missed]
             for missed in missed_ids)
        ]
        assert all(len(x) > 0 for x in false_negatives), 'False negatives must be properly found'

        if self.examples:
            for name, examples in (('fn', false_negatives), ('fp', false_positives), ('mix', partial_mismatch)):
                for words, lemma in examples:
                    present_mwes = [
                        (sense_data.lemma, tuple(str(w.idx) for w in words))
                        for sense_data, words in sentence.get_mwe_groups()
                    ]
                    index_str = f'({",".join(str(w.idx) for w in words)})'
                    cleaned_text = re.sub(r'\s', ' ', sentence.original_text)
                    outstr = f'{name}\t{lemma}\t{index_str}\t{cleaned_text}\t{present_mwes}\n'
                    self.example_file.write(outstr)

        if self.printing:
            for predicted_label, gold_label, word in zip(predicted_labels, gold_labels, sentence):
                gold_lemma = '_' if not word.mwe_sense_data else word.mwe_sense_data.lemma
                predicted_lemma = '_' if not hasattr(word, 'mwe_candidate') else word.mwe_candidate.mwe_data.lemma
                output_row = [word.form, word.word_sense_data.pos, predicted_lemma, gold_lemma,
                              str(predicted_label), str(gold_label)]
                self.file.write('\t'.join(output_row) + '\n')
            self.file.write('\n\n')

        return EvalLabels(predicted_labels, gold_labels, sent_results.eval_data, sent_results,
                          false_negatives, false_positives, partial_mismatch)

    def _yield_labels(self) -> Iterable[Tuple[int, int]]:
        for sentence in tqdm(self.data, desc='MWE Eval'):
            labels = self.labels_for_sentence(sentence)

            for predicted_label, gold_label in zip(labels.predicted, labels.gold):
                yield predicted_label, gold_label

    def __call__(self) -> Tuple[EvalStats, EvalStats, Dict[str, Dict[str, Dict]]]:
        predicted_labels = []
        gold_labels = []
        pipeline_eval_labels = defaultdict(lambda: defaultdict(list))

        for sentence in tqdm(self.data, desc='MWE Eval'):
            labels = self.labels_for_sentence(sentence)
            predicted_labels.extend(labels.predicted)
            gold_labels.extend(labels.gold)
            for filter_classname, eval_data_list in labels.eval_data.items():
                for eval_data in eval_data_list:
                    for field in fields(FilterEvalData):
                        fieldname = field.name
                        pipeline_eval_labels[filter_classname][fieldname].extend(getattr(eval_data, fieldname))

        if self.printing:
            print(f'Wrote to {self.printing_filename}')
            self.file.close()

        if self.examples:
            print(f'Wrote examples to {self.examples}')
            self.example_file.close()

        pipeline_eval_stats = {}
        for filter_classname in pipeline_eval_labels:
            # TODO: currently not doing anything with MWE labels, because we want to handle the partial
            # match case differently somehow (label is 2 for partial match)
            word_hyp = pipeline_eval_labels[filter_classname]['word_hyp_labels']
            word_gold = pipeline_eval_labels[filter_classname]['word_gold_labels']
            mwe_hyp_labels = pipeline_eval_labels[filter_classname]['mwe_hyp_labels']
            pipeline_eval_stats[filter_classname] = {
                'word': classification_report(word_gold, word_hyp, output_dict=True),
                'mwe_hyp_pos_rate': sum(mwe_hyp_labels) / len(mwe_hyp_labels)
            }

        return self._compute_aligned_eval_output(gold_labels, predicted_labels), \
               self._compute_binary_eval_output(gold_labels, predicted_labels), pipeline_eval_stats


class KulkarniMWEEvaluator(MWEEvaluator):
    def __init__(self, data: List[TrainingSentence], model: Optional[ContextDictionaryBiEncoder] = None, **kwargs):
        if kwargs.get('no_filters', False):
            filters = []
        else:
            filters = [OrderedOnly(), MaxGappiness(3)]
            if model is not None:
                filters.append(ModelOutputIsMWE())

        if kwargs.get('no_resolver', False):
            resolver = None
        else:
            resolver = KulkarniResolverWrapper(ModelScoreResolver() if model else LessGappyLongerResolver())

        pipeline = MWEPipeline(
            index=MWEIndex.get_or_build_index(Path('kulkarni_index.db'),
                                              input_data=Path("data/mweindex_wordnet3.0_semcor1.6.data"),
                                              in_memory=True),
            filters=filters,
            resolver=resolver,
            model=model,
            additional_detectors=[KulkarniPropNounDetector()],
            detect_with_form=True
        )
        super(KulkarniMWEEvaluator, self).__init__(pipeline, data, **kwargs)


class CuptMWEEvaluator(MWEEvaluator):

    def __init__(self, data: List[TrainingSentence], model: Optional[ContextDictionaryBiEncoder] = None, **kwargs):
        if kwargs.get('no_filters', False):
            filters = [SpecificPOSOnly({'v'})]
        else:
            # Note that non-verb MWEs are filtered out
            filters = [OrderedOnly(), MaxGappiness(3), SpecificPOSOnly({'v'})]
            if model is not None:
                filters.append(ModelOutputIsMWE())

        if kwargs.get('no_resolver', False):
            resolver = None
        else:
            resolver = ModelScoreResolver() if model else LessGappyLongerResolver()

        pipeline = MWEPipeline(
            index=MWEIndex.get_or_build_index(Path('temp_index.db'), in_memory=True),
            filters=filters,
            resolver=resolver,
            model=model
        )
        super(CuptMWEEvaluator, self).__init__(pipeline, data, **kwargs)

    def cupt_lines(self, disable_printing: bool = False) -> Iterable[str]:
        column_names = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc', 'mwe']
        label_iter = iter(self._yield_labels())

        gold_labels = []
        predicted_labels = []

        yield '# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n'
        for sentence in self.data:
            yield f'# source_sent_id = . . {sentence.metadata["id"]}\n'
            yield f'# text = {sentence.original_text}\n'
            for word in sentence:
                predicted_label, gold_label = next(label_iter)
                gold_labels.append(gold_label)
                predicted_labels.append(predicted_label)
                row_data = [word.word_sense_data.metadata[colname] for colname in column_names]
                row_data[-1] = str(predicted_label) if predicted_label != 0 else '*'
                yield '\t'.join(row_data) + '\n'

            yield '\n'

        if not disable_printing:
            print(self._compute_aligned_eval_output(gold_labels, predicted_labels))

    def streusle_lines(self, disable_printing: bool = False) -> Iterable[str]:
        column_names = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc', 'mwe',
                        'lexcat', 'lexlemma', 'supersense', 'second_supersense', 'weak_mwe',
                        'wcat_unused', 'weakmwe_lemma', 'bio_tag']

        mwe_index = column_names.index('mwe')
        lexlemma_index = column_names.index('lexlemma')
        bio_tag_index = column_names.index('bio_tag')
        weak_mwe_index = column_names.index('weak_mwe')
        weak_mwe_lemma_index = column_names.index('weakmwe_lemma')
        lexcat_index = column_names.index('lexcat')
        lemma_index = column_names.index('lemma')
        upos_index = column_names.index('upos')
        supersense_index = column_names.index('supersense')

        gold_labels = []
        predicted_labels = []

        for sentence in tqdm(self.data, 'streusle MWE out'):
            # newdoc id = reviews-001961
            # sent_id = reviews-001961-0001
            # newpar id = reviews-001961-p0001
            # text = Buyer Beware!!
            # streusle_sent_id = ewtb.r.001961.1
            # mwe = Buyer~Beware !!
            yield "# newdoc id = <placeholder>\n"
            yield f'# sent_id = {sentence.metadata["id"]}\n'
            yield "# newpar id = <placeholder>\n"
            yield f'# text = {sentence.original_text}\n'
            yield f'# streusle_sent_id = <placeholder>\n'
            yield "# mwe = <placeholder>\n"
            label_data = self.labels_for_sentence(sentence)
            mwe_by_idx = {
                word.idx: mwe_result for mwe_result in label_data.pipeline_results
                for word in mwe_result.words
            }

            for word, predicted_label, gold_label in zip(sentence, label_data.predicted, label_data.gold):
                gold_labels.append(gold_label)
                predicted_labels.append(predicted_label)
                metadata = word.word_sense_data.metadata

                row_data = [metadata[colname] for colname in column_names]

                #  we mark everything as strong MWES, not ideal
                if word.idx in mwe_by_idx:
                    supersense = f'-{metadata["supersense"]}' if metadata['supersense'] != '_' else ''
                    lexcat = f'-{metadata["lexcat"]}' if metadata['lexcat'] != '_' else ''
                    mwe = mwe_by_idx[word.idx]
                    index_in_mwe = mwe.words.index(word) + 1
                    index_of_mwe = label_data.pipeline_results.index(mwe) + 1
                    if index_in_mwe == 1:
                        bio_tag = 'B' + lexcat + supersense
                        lexlemma = ' '.join(mwe.mwe_data.lemma.split('_'))
                    else:
                        bio_tag = 'I_'
                        lexlemma = '_'

                    mwe_tag = f'{index_of_mwe}:{index_in_mwe}'
                    row_data[lexlemma_index] = lexlemma
                else:
                    if row_data[-1].startswith('O'):
                        yield '\t'.join(row_data) + '\n'
                        continue

                    mwe_tag = '_'
                    lexlemma_default = row_data[lexlemma_index]
                    if ' ' in lexlemma_default:
                        row_data[lexlemma_index] = lexlemma_default.split()[0]
                    else:
                        row_data[lexlemma_index] = row_data[lemma_index]

                    row_data[lexcat_index] = {'NOUN': 'N',
                                              'PROPN': 'N',
                                              'VERB': 'V',
                                              'ADP': 'DISC',
                                              'ADV': 'DISC',
                                              'SCONJ': 'DISC',
                                              'PART': 'POSS'}.get(row_data[upos_index],
                                                                  row_data[upos_index].split('.')[0])
                    if row_data[lexcat_index] in ('N', 'V', 'P', 'INF.P', 'PP', 'POSS', 'PRON.POSS'):
                        row_data[supersense_index] = ''
                    else:
                        row_data[supersense_index] = '??'
                        supersense = f'-{row_data[supersense_index]}' if row_data[supersense_index] != '_' else ''

                    lexcat = f'-{row_data[lexcat_index]}' if row_data[lexcat_index] != '_' else ''
                    bio_tag = 'O' + lexcat + supersense

                row_data[mwe_index] = mwe_tag
                row_data[bio_tag_index] = bio_tag
                row_data[weak_mwe_index] = '_'
                row_data[weak_mwe_lemma_index] = '_'

                yield '\t'.join(row_data) + '\n'

            yield '\n'

        if not disable_printing:
            print(self._compute_aligned_eval_output(gold_labels, predicted_labels))


class DimSumMWEEvaluator(MWEEvaluator):

    def __init__(self, data: List[TrainingSentence], model: Optional[ContextDictionaryBiEncoder] = None, **kwargs):
        # dimsum doesn't allow discontiguous MWEs, so just ban everything with gaps
        if kwargs.get('no_filters', False):
            filters = []
        else:
            filters = [OrderedOnly(), MaxGappiness(3)]
            if model is not None:
                filters.append(ModelOutputIsMWE())

        if kwargs.get('no_resolver', False):
            resolver = None
        else:
            resolver = ModelScoreResolver() if model else LessGappyLongerResolver()

        pipeline = MWEPipeline(
            index=MWEIndex.get_or_build_index(Path('temp_index.db'), in_memory=True),
            filters=filters,
            resolver=resolver,
            model=model,
            # additional_detectors=[DimSumSequentialNounDetector()] # Detect MW proper nouns
        )
        super(DimSumMWEEvaluator, self).__init__(pipeline, data, **kwargs)

    # Enable reading the test data in the PARSEME format
    def cupt_lines(self, disable_printing: bool = False) -> Iterable[str]:
        column_names = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc', 'mwe']
        label_iter = iter(self._yield_labels())

        gold_labels = []
        predicted_labels = []

        yield '# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n'
        for sentence in self.data:
            yield f'# source_sent_id = . . {sentence.metadata["id"]}\n'
            yield f'# text = {sentence.original_text}\n'
            for word in sentence:
                predicted_label, gold_label = next(label_iter)
                gold_labels.append(gold_label)
                predicted_labels.append(predicted_label)
                row_data = [word.word_sense_data.metadata[colname] for colname in column_names]
                row_data[-1] = str(predicted_label) if predicted_label != 0 else '*'
                yield '\t'.join(row_data) + '\n'

            yield '\n'

        if not disable_printing:
            print(self._compute_aligned_eval_output(gold_labels, predicted_labels))

    def dimsum_lines(self, disable_printing: bool = False) -> Iterable[str]:
        column_names = ['idx', 'word', 'lemma', 'pos', 'mwe_tag', 'parent_idx', 'unused', 'supersense', 'sent_id']
        gold_labels = []
        predicted_labels = []

        mwe_idx = column_names.index('mwe_tag')
        parent_idx_idx = column_names.index('parent_idx')
        supersense_idx = column_names.index('supersense')

        for sentence in tqdm(self.data, 'dimsum MWE out'):
            label_data = self.labels_for_sentence(sentence)
            mwe_by_idx = {
                word.idx: mwe_result for mwe_result in label_data.pipeline_results
                for word in mwe_result.words
            }

            gapped_indices = set()
            for mwe_result in label_data.pipeline_results:
                mwe_indices = set(w.idx for w in mwe_result.words)
                start = min(mwe_indices)
                end = max(mwe_indices)
                gaps = set(range(start, end)) - set(mwe_indices)
                gapped_indices = gapped_indices | gaps

            for word, predicted_label, gold_label in zip(sentence, label_data.predicted, label_data.gold):
                gold_labels.append(gold_label)
                predicted_labels.append(predicted_label)
                metadata = word.word_sense_data.metadata

                row_data = [metadata[colname] for colname in column_names]

                if word.idx in mwe_by_idx:
                    mwe = mwe_by_idx[word.idx]
                    index_in_mwe = mwe.words.index(word)
                    if mwe.words.index(word) == 0:
                        row_data[mwe_idx] = 'b' if word.idx in gapped_indices else 'B'
                        row_data[parent_idx_idx] = '0'
                    else:
                        row_data[mwe_idx] = 'i' if word.idx in gapped_indices else 'I'
                        row_data[parent_idx_idx] = mwe.words[index_in_mwe - 1].word_sense_data.metadata['idx']
                        row_data[supersense_idx] = ''

                else:
                    row_data[mwe_idx] = 'o' if word.idx in gapped_indices else 'O'
                    row_data[parent_idx_idx] = '0'

                yield '\t'.join(row_data) + '\n'

            yield '\n'

        if not disable_printing:
            print(self._compute_aligned_eval_output(gold_labels, predicted_labels))


class MWEEvalData(Enum):
    KULKARNI = 'kulkarni'
    CUPT_TRAIN = 'cupt_train'
    CUPT_TEST = 'cupt_test'
    STREUSLE_DEV = 'streusle_dev'
    STREUSLE_TEST = 'streusle_test'
    DIMSUM_TRAIN = 'dimsum_train'
    DIMSUM_TEST = 'dimsum_test'

    KULKARNI_SAMPLE = 'kulkarni_sample'
    CUPT_SAMPLE = 'cupt_sample'
    DIMSUM_SAMPLE = 'dimsum_sample'

    COAM_TRAIN = 'coam_train'
    COAM_TEST = 'coam_test'

    def get_evaluator(self, model: Optional[ContextDictionaryBiEncoder], max_count: Optional[int] = None,
                      **kwargs) -> MWEEvaluator:
        dataset_file, evaluator_class = DATASET_TUPLE_DICT[self]

        data = read_training_sentences(dataset_file, None)
        if max_count is not None:
            data = islice(data, max_count)
        else:
            data = list(data)

        return evaluator_class(data, model, **kwargs)


DATASET_TUPLE_DICT = {
    MWEEvalData.CUPT_TRAIN: (DATA_DIR / 'cupt_train.jsonl', CuptMWEEvaluator),
    MWEEvalData.CUPT_TEST: (DATA_DIR / 'cupt_test.jsonl', CuptMWEEvaluator),
    MWEEvalData.KULKARNI: (DATA_DIR / 'kulkarni.jsonl', KulkarniMWEEvaluator),
    MWEEvalData.KULKARNI_SAMPLE: (
        DATA_DIR / 'kulkarni_sample.jsonl',
        KulkarniMWEEvaluator,
    ),
    MWEEvalData.CUPT_SAMPLE: (DATA_DIR / 'cupt_train_dev.jsonl', CuptMWEEvaluator),
    MWEEvalData.STREUSLE_DEV: (DATA_DIR / 'streusle_dev.jsonl', CuptMWEEvaluator),
    MWEEvalData.STREUSLE_TEST: (DATA_DIR / 'streusle_test.jsonl', CuptMWEEvaluator),
    MWEEvalData.DIMSUM_TEST: (DATA_DIR / 'dimsum_test.jsonl', DimSumMWEEvaluator),
    MWEEvalData.DIMSUM_TRAIN: (DATA_DIR / 'dimsum_train.jsonl', DimSumMWEEvaluator),
    MWEEvalData.DIMSUM_SAMPLE: (
        DATA_DIR / 'dimsum_train_dev.jsonl',
        DimSumMWEEvaluator,
    ),
    MWEEvalData.COAM_TRAIN: (
        Path('../data/50_mweaswsd/coam_train.tanner.jsonl'),
        DimSumMWEEvaluator,
    ),
    MWEEvalData.COAM_TEST: (
        Path('../data/50_mweaswsd/coam_test.tanner.jsonl'),
        DimSumMWEEvaluator,
    ),
}
