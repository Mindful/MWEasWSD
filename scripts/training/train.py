import re
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from distutils.util import strtobool
import sys
import shutil
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from resolve.common.data import WordnetDefinitionLookup
from resolve.common.util import yaml_to_dict
from resolve.training import DRY_RUN_DATA, DEFAULT_PLUS_OMSTI_DATA, DEFAULT_DATA, CUPT_DATA, DIMSUM_DATA, \
    MIXED_FINETUNE, MIXED_TRAIN, WSD_PATH
from resolve.training.data import DefinitionMatchingDataset, DefinitionMatchingFixedCountBatchDataset, \
    DefinitionMatchingLoader, wsd_candidates, use_only_candidate_wordnet
from resolve.model.pl_module import ContextDictionaryBiEncoder
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from time import time
from random import Random
from nltk.corpus import words
import git
import torch

from resolve.training.mwe_eval import MWEEvalData

data_loading = {
    cls.__name__: cls for cls in (DefinitionMatchingDataset, DefinitionMatchingFixedCountBatchDataset)
}

data_choices = {
    'dry_run': DRY_RUN_DATA,
    'default_plus_omsti': DEFAULT_PLUS_OMSTI_DATA,
    'default': DEFAULT_DATA,
    'cupt': CUPT_DATA,
    'dimsum': DIMSUM_DATA,
    'mixed_finetune': MIXED_FINETUNE,
    'mixed_train': MIXED_TRAIN
}

language_choices = [
    'eng',
    'jpn'
]


def str_to_bool(s: str) -> bool:
    return bool(strtobool(s))


def generate_run_name() -> str:
    # we have to seed this separately, because otherwise Lightning's seed_everything means we always get the same word
    rand = Random(time())
    random_word = rand.sample(words.words(), 1)[0]
    return f'{random_word}-{int(time())}'


def load_model(args: Namespace) -> ContextDictionaryBiEncoder:
    accepted_args = {"encoder", "lr", "weight_decay", "dropout", "head_config", "definition_encoder"}
    if args.load_model is not None:
        print(f'Loading model from {args.load_model}, ignoring encoder and head config input arguments')
        accepted_args = accepted_args - {"encoder", "definition_encoder", "head_config"}

    model_args = {
        k: v for k, v in vars(args).items()
        if k in accepted_args
    }

    if args.load_model is None:
        if model_args['definition_encoder'] is None:
            print(f'Definition encoder defaulting to the same value as main encoder: {model_args["encoder"]}')
            model_args['definition_encoder'] = model_args['encoder']

        return ContextDictionaryBiEncoder(**model_args)
    else:
        return ContextDictionaryBiEncoder.load_from_checkpoint(args.load_model, **model_args)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--head_config', type=yaml_to_dict, default="configs/default_mwe_neg_cross_product.yaml",
                        required=False)
    parser.add_argument('--definition_encoder', type=str, required=False)
    parser.add_argument('--definition_language', type=str, choices=language_choices, default='eng')
    parser.add_argument('--data_loading', type=str, choices=list(data_loading.keys()),
                        default=DefinitionMatchingFixedCountBatchDataset.__name__)
    parser.add_argument('--mwe_processing', type=str_to_bool, default='False')
    parser.add_argument('--wsd_processing', type=str_to_bool, default='True')
    parser.add_argument('--train_data_suffix', type=str, default=None)
    parser.add_argument('--data', type=str, choices=list(data_choices.keys()), default='default')
    parser.add_argument('--include_single_definitions', type=str_to_bool, default='False')
    parser.add_argument('--swa', type=str_to_bool, default='True')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--run_name', default=generate_run_name())
    parser.add_argument('--use_wandb', type=str_to_bool, default='True')
    parser.add_argument('--skip_mwe_pipeline_eval', type=str_to_bool, default='False')
    parser.add_argument('--load_model', type=str, required=False)
    parser.add_argument('--checkpoint_metric', type=str, default=ContextDictionaryBiEncoder.val_f1)
    parser.add_argument('--mwe_eval_pipelines', type=MWEEvalData, nargs='+',
                        default=[MWEEvalData.DIMSUM_SAMPLE, MWEEvalData.CUPT_SAMPLE])
    parser.add_argument('--run_final_mwe_eval', type=str_to_bool, default='False')
    parser.add_argument('--save_model_to_wandb', type=str_to_bool, default='False')
    parser.add_argument('--remove_checkpoint_dir', type=str_to_bool, default='False')
    parser.add_argument('--upsample_train_data', type=int, nargs='+')
    parser.add_argument('--limit_key_candidates', type=str_to_bool, default='True')
    parser.add_argument('--checkpoint_save_count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1337)

    known_args, _ = parser.parse_known_args()
    hyperparameters = (set(vars(known_args)) | {
        'max_epochs', 'accumulate_grad_batches', 'gradient_clip_val', 'gpus'
    }) - {'run_name'}  # get important args before we add pytorch-lightning args

    set_values = {x[2:].split('=')[0] for x in sys.argv if x.startswith('--')}
    untracked_args = set_values - hyperparameters
    if len(untracked_args) > 0:
        print(f'Warning: found arguments for normally untracked hyperparameters {untracked_args}')
        print('These arguments will be added to tracked hyperparameters')
        hyperparameters.update(untracked_args)

    if known_args.mwe_processing and not known_args.train_data_suffix:
        print('WARNING: You are running with MWE processing enabled but on vanilla WSD data; '
              'there will be no negative MWE examples')

    Trainer.add_argparse_args(parser)
    args = Trainer.parse_argparser(parser.parse_args())
    seed_everything(args.seed)
    print(f'--------BEGINNING RUN {args.run_name}--------')
    if args.limit_key_candidates:
        use_only_candidate_wordnet()

    model = load_model(args)
    context_tokenizer = AutoTokenizer.from_pretrained(model.context_encoder.name_or_path)
    definition_tokenizer = AutoTokenizer.from_pretrained(model.definition_encoder.name_or_path)

    datasets = DefinitionMatchingLoader(data_choices[args.data], context_tokenizer, args.batch_size,
                                        data_loading[args.data_loading],
                                        allow_single_def=args.include_single_definitions,
                                        definition_language=args.definition_language,
                                        definition_tokenizer=definition_tokenizer,
                                        include_mwe=args.mwe_processing,
                                        include_wsd=args.wsd_processing,
                                        train_data_suffix=args.train_data_suffix,
                                        upsample_train=args.upsample_train_data)

    if args.mwe_processing and not args.skip_mwe_pipeline_eval:
        mwe_pipelines = args.mwe_eval_pipelines
    else:
        mwe_pipelines = []

    model.setup_for_train_eval(datasets.manager, mwe_eval=mwe_pipelines)

    callbacks = []
    if args.enable_checkpointing:
        checkpoint_dir = f"checkpoints/{args.run_name}/"
        callbacks.append(ModelCheckpoint(
            monitor=args.checkpoint_metric,
            dirpath=checkpoint_dir,
            filename="ep{epoch:02d}_{" + args.checkpoint_metric + ":.2f}f1",
            save_top_k=args.checkpoint_save_count,
            mode="max",
            auto_insert_metric_name=False
        ))
    else:
        assert not args.run_final_mwe_eval, "Can't run final MWE eval with checkpointing disabled"

    if args.swa:
        callbacks.append(StochasticWeightAveraging(args.lr))

    if args.use_wandb:
        if args.save_model_to_wandb:
            assert args.enable_checkpointing, 'Can only save to wandb if checkpointing enabled'

        wandb_logger = WandbLogger(name=args.run_name, log_model=args.save_model_to_wandb)
        wandb_logger.experiment.config['dataset_stats'] = datasets.get_summary_stats_dict()
        wandb_logger.experiment.config['head_config'] = args.head_config
        hyperparams_to_log = {key: val for key, val in vars(args).items() if key in hyperparameters}
        hyperparams_to_log['git_sha'] = git.Repo(search_parent_directories=True).head.object.hexsha
        wandb_logger.experiment.config.update(hyperparams_to_log)
        for f1_metric_name in ContextDictionaryBiEncoder.f1_metrics:
            wandb_logger.experiment.define_metric(f1_metric_name, summary='max', step_metric='epoch')
        for loss_metric_name in ContextDictionaryBiEncoder.loss_metrics:
            wandb_logger.experiment.define_metric(loss_metric_name, summary='min', step_metric='epoch')
        loggers = [wandb_logger]
    else:
        assert not args.save_model_to_wandb, 'Cannot save model to wandb if not using wandb'
        loggers = [CSVLogger("logs", name=args.run_name)]

    if not torch.cuda.is_available():
        args.accelerator = 'cpu'
        args.gpus = 0

    trainer = Trainer.from_argparse_args(args, logger=loggers, callbacks=callbacks)

    trainer.fit(model, datasets)

    if args.run_final_mwe_eval:
        print('Running final MWE eval')
        model = load_best_model(checkpoint_dir)
        model.setup_for_train_eval(datasets.manager)
        if torch.cuda.is_available():
            model.cuda()

        for eval_data in [MWEEvalData.DIMSUM_TRAIN, MWEEvalData.CUPT_TRAIN, MWEEvalData.KULKARNI]:
            evaluator = eval_data.get_evaluator(model)
            aligned_results, binary_results, pipeline_results = evaluator()
            eval_results = {
                    **{
                        f'{eval_data.value}/aligned/{key}': val for key, val in asdict(aligned_results).items()
                    },
                    **{
                        f'{eval_data.value}/binary/{key}': val for key, val in asdict(binary_results).items()
                    }
            }
            if args.use_wandb:
                for key, val in eval_results.items():
                    wandb_logger.experiment.summary[key] = val
            else:
                print(eval_results)

    if args.remove_checkpoint_dir:
        shutil.rmtree(checkpoint_dir)



def load_best_model(checkpoint_dir: str) -> ContextDictionaryBiEncoder:
    candidates_map = {
        int(next(re.finditer(r'\d+?(?=f1)', p.name)).group(0)) : p
        for p in Path(checkpoint_dir).glob('*.ckpt')
    }
    path = candidates_map[max(candidates_map.keys())]
    print('Loading best model:', path.absolute())
    return ContextDictionaryBiEncoder.load_from_checkpoint(str(path.absolute()))


if __name__ == '__main__':
    main()
