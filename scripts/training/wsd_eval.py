from pathlib import Path
from argparse import ArgumentParser
from subprocess import run

from resolve.common.data import WordnetDefinitionLookup
from resolve.common.util import flatten
from train import language_choices, str_to_bool
from resolve.training import WSD_PATH
from resolve.training.data import TrainingDefinitionManager, DefinitionMatchingDataset, wsd_candidates, \
    use_only_candidate_wordnet
from resolve.model.pl_module import ContextDictionaryBiEncoder
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader


scorer_path = WSD_PATH / 'Evaluation_Datasets/Scorer.java'
WSD_OFFICIAL = 'wsd_official'
LIGHTNING = 'lightning'


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--scoring', type=str, choices=[LIGHTNING, WSD_OFFICIAL], default=WSD_OFFICIAL)
    parser.add_argument('--definition_language', type=str, choices=language_choices, default='eng')
    parser.add_argument('--mwe_processing', type=str_to_bool, default='False')
    parser.add_argument('--limit_key_candidates', type=str_to_bool, default='True')

    args = parser.parse_args()
    if args.limit_key_candidates:
        use_only_candidate_wordnet()

    model = ContextDictionaryBiEncoder.load_from_checkpoint(str(args.model))
    context_tokenizer = AutoTokenizer.from_pretrained(model.context_encoder.name_or_path)
    definition_tokenizer = AutoTokenizer.from_pretrained(model.definition_encoder.name_or_path)
    manager = TrainingDefinitionManager(context_tokenizer, definition_tokenizer,
                                        allow_single_def=True, def_language=args.definition_language,
                                        mwe_training=False)
    data = DefinitionMatchingDataset([args.data], manager, args.batch_size)
    model.setup_for_train_eval(manager)

    with torch.no_grad():
        if args.scoring == LIGHTNING:
            trainer = Trainer(gpus=1 if torch.cuda.is_available() else None)
            dataloader = DataLoader(data, batch_size=None, collate_fn=data.collate_into_batch)
            results = trainer.validate(model, dataloader)
            print('Results -------')
            print(results)
            print('---------------')
        elif args.scoring == WSD_OFFICIAL:
            data_path = data.dataset_paths[0]
            gold_path = data_path / '{}.gold.key.txt'.format(data_path.name)

            assert gold_path.exists(), f'Gold path must exist but did not find {gold_path}'

            if not scorer_path.exists():
                raise RuntimeError("Didn't find Scorer.java, it probably needs to be compiled")
            model.eval()
            if torch.cuda.is_available():
                model.cuda()

            all_words = []
            all_labels = []
            for sentences_for_batch in tqdm(data, 'processing batches'):
                model_input = data.collate_into_batch(sentences_for_batch)
                words = data.words_for_batch(sentences_for_batch, model_input)

                if torch.cuda.is_available():
                    model_input = {
                        k: v.cuda() for k, v in model_input.items() if k != 'word_sense_labels'
                    }
                else:
                    del model_input['word_sense_labels']

                output = model(model_input)
                labels = list(x.label_pred for x in flatten(output.word_outputs))

                all_words.extend([w for wordlist in words for w in wordlist])
                all_labels.extend(labels)
                assert len(all_words) == len(all_labels), 'Words and labels must stay aligned'

        print('Writing output')
        output_filename = Path('out.txt')
        outputs = list(zip(all_words, all_labels))
        outputs.sort(key=lambda x: x[0].word_sense_data.metadata['wsd_id'])

        with output_filename.open('w') as f:
            for word, label in outputs:
                wsd_id = word.word_sense_data.metadata["wsd_id"]
                pred_key = word.word_sense_data.label_to_key(label)
                f.write(f'{wsd_id} {pred_key}\n')

        print('Running scorer -------')
        run(['java', scorer_path, str(gold_path.absolute()), output_filename])
        print('----------------------')

    print('Done')


if __name__ == '__main__':
    main()
