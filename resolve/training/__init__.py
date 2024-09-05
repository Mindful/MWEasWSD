from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = BASE_DIR / 'data'
WSD_PATH = DATA_DIR / 'WSD_Evaluation_Framework/'
eval_path = WSD_PATH / 'Evaluation_Datasets/'
test_dir = BASE_DIR / 'test'
DEFAULT_DATA = {
    'train': [WSD_PATH / 'Training_Corpora/SemCor'],
    'dev': [eval_path / 'semeval2007'],
    'test': [eval_path / 'semeval2013', eval_path / 'semeval2015',
             eval_path / 'senseval2', eval_path / 'senseval3']
}
DEFAULT_PLUS_OMSTI_DATA = {
    'train': [WSD_PATH / 'Training_Corpora/SemCor+OMSTI'],
    'dev': [eval_path / 'semeval2007'],
    'test': [eval_path / 'semeval2013', eval_path / 'semeval2015',
             eval_path / 'senseval2', eval_path / 'senseval3']
}
DRY_RUN_DATA = {
    'train': [DATA_DIR / 'dryrun'],
    'dev': [DATA_DIR / 'dryrun'],
    'test': [DATA_DIR / 'dryrun']
}
# A copy of dry run data checked into git so it doesn't change
TEST_DATA = {
    'train': [test_dir / 'testdata'],
    'dev': [test_dir / 'testdata'],
    'test': [test_dir / 'testdata']
}
CUPT_DATA = {
    'train': [DATA_DIR / 'cupt_train_train'],
    'dev': [DATA_DIR / 'dryrun'],  # not used
    'test': [DATA_DIR / 'dryrun']  # not used
}
DIMSUM_DATA = {
    'train': [DATA_DIR / 'dimsum_train_train'],
    'dev': [DATA_DIR / 'dryrun'],  # not used
    'test': [DATA_DIR / 'dryrun']  # not used
}

MIXED_FINETUNE = {
    'train': [DATA_DIR / 'cupt_train_train', DATA_DIR / 'dimsum_train_train'],
    'dev': [DATA_DIR / 'dryrun'],  # not used
    'test': [DATA_DIR / 'dryrun']  # not used
}
MIXED_TRAIN = {
    'train': [WSD_PATH / 'Training_Corpora/SemCor'] + MIXED_FINETUNE['train'],
    'dev': [eval_path / 'semeval2007'],
    'test': [eval_path / 'semeval2013', eval_path / 'semeval2015',
             eval_path / 'senseval2', eval_path / 'senseval3']
}

COAM_FINETUNE = {
    'train': [DATA_DIR / 'coam_train'],
    'dev': [DATA_DIR / 'dryrun'],  # not used
    'test': [DATA_DIR / 'dryrun'],  # not used
}
