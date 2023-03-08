from argparse import ArgumentParser
from pathlib import Path
from jsonlines import open as open_jsonl
from tqdm import tqdm


from resolve.training.data import read_training_sentences, fast_linecount, SenseData
from annotation_candidates import get_pipeline
from resolve.training.mwe_preproc.common import ANNOTATION_KEY
from resolve.training.mwe_preproc.new import get_new_candidates

"""This script was used to update annotations form an old format like this:

export SEMCOR_PATH=data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.fixed.jsonl 
python scripts/preprocessing/update_annotations.py annotations/a_annotations.jsonl $SEMCOR_PATH  --mwe_vocab mwe_vocab.txt
python scripts/preprocessing/update_annotations.py annotations/b_annotations.jsonl $SEMCOR_PATH  --mwe_vocab mwe_vocab.txt

"""



def main():
    parser = ArgumentParser()
    parser.add_argument('annotations', type=Path)
    parser.add_argument('sentences', type=Path)
    parser.add_argument('--mwe_vocab', type=Path, required=True,
                        help='MWE Vocab file, should be the same as used for '
                             'original candidate generation (see readme)')

    args = parser.parse_args()
    output_path = args.annotations.parent / ('updated_' + args.annotations.name)

    training_sentences = list(read_training_sentences(args.sentences, None))
    sentences_by_text = {
        sent.original_text: sent for sent in training_sentences
    }
    total = fast_linecount(args.annotations)

    pipeline = get_pipeline(args.mwe_vocab)
    output_entires = []
    dropped_annotations = 0
    for prodigy_entry in tqdm(open_jsonl(args.annotations), total=total, desc=args.annotations.name):
        sentence_text = prodigy_entry['text']
        sentence = sentences_by_text[sentence_text]
        sense_data = SenseData.from_json(prodigy_entry['_sense_data'], None, True)
        if prodigy_entry['answer'] == 'accept':
            assert sense_data.gold_sense is not None

        candidates = get_new_candidates(pipeline, sentence, None)
        try:
            corresponding_annotation = next(c for c in candidates if c.sense_data.lemma == sense_data.lemma)
            corresponding_annotation.sense_data.gold_sense = sense_data.gold_sense
            corresponding_annotation.sense_data.pos = sense_data.pos
        except StopIteration:
            dropped_annotations += 1
            continue

        prodigy_entry[ANNOTATION_KEY] = corresponding_annotation.to_json()
        output_entires.append(prodigy_entry)

    print('Had to drop', dropped_annotations, 'annotations')
    print('Writing output to', output_path)
    with open_jsonl(output_path, 'w') as output_file:
        output_file.write_all(output_entires)

    print('Done, wrote', len(output_entires), 'annotations')


if __name__ == '__main__':
    main()
