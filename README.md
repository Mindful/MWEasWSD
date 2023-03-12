# MWE as WSD
Repo for the paper `MWE as WSD: Solving Multi-Word Expression Identification with Word
Sense Disambiguation`.

## Installation
```shell
cd MWEasWSD
pip install -e .
```

## Data

Download the data (requires [gdown](https://github.com/wkentaro/gdown) to download some pre-processed evaluation data,
but this data is not strictly necessary and can be skipped). 

```shell
cd data && ./get-data.sh
```

### Preprocessing

Generate `.jsonl` files and split data. 
```shell
# Convert the data into .jsonl files
python scripts/preprocessing/wsd_xml.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor
cp data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/all.data.xml
cp data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/all.gold.key.txt
for d in "data/WSD_Evaluation_Framework/Evaluation_Datasets/"*"/";
  do python scripts/preprocessing/wsd_xml.py "$d"
done

python scripts/preprocessing/connl_u.py --data_type cupt data/parseme_mwe/EN/train.cupt data/cupt_train.jsonl
python scripts/preprocessing/connl_u.py --data_type cupt data/parseme_mwe/EN/test.cupt data/cupt_test.jsonl

python scripts/preprocessing/dimsum.py data/dimsum-data/dimsum16.test data/dimsum_test.jsonl
python scripts/preprocessing/dimsum.py data/dimsum-data/dimsum16.train data/dimsum_train.jsonl


# Generate data using scripts/preprocessing/PrintData.java if necessary (see comments in file for instructions)
python scripts/preprocessing/kulkarni.py data/kulkarni_finlayson_data.txt

# Samples
python scripts/preprocessing/random_sample.py data/kulkarni.jsonl
python scripts/preprocessing/random_split.py data/cupt_train.jsonl
python scripts/preprocessing/random_split.py data/dimsum_train.jsonl
python scripts/preprocessing/fix_existing_mwes.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.jsonl 
```

Apply annotations, add automatic negatives
```shell
python scripts/preprocessing/apply_mwe_annotations.py data/annotations.jsonl data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.fixed.jsonl
python scripts/preprocessing/auto_add_negative_mwes.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.fixed.annotated.jsonl --target_percent 0.55
```

## Model download
A model pre-trained on the modified SemCor data can be downloaded [here](https://drive.google.com/file/d/1Os2LMXoqWu1JV8ibr66KpTACdwBprf0d/view?usp=share_link).
Unless otherwise specified, models are provided under the same license as the code in this repository. 

## Training

To replicate our top run:
```shell
python scripts/training/train.py \
--max_epochs 15 \
--batch_size 16 \
--accumulate_grad_batches 2 \
--gpus 1 \
--swa true \
--gradient_clip_val 1.0 \
--lr 0.00001 \
--run_name replicate-top \
--encoder bert-base-uncased \
--enable_checkpointing false \
--mwe_processing true \
--train_data_suffix fixed.annotated.autoneg
```


### Finetune on DiMSUM/PARSEME data
```shell
export BASE_MODEL=checkpoints/replicate-top/ep14_0.73f1.ckpt # set this to the base model you want to use

# add candidates using the model we want to finetune, so it learns from its own mistakes
python scripts/preprocessing/auto_add_negative_mwes.py data/dimsum_train_train.jsonl \
--target_percent 1.0 --pipeline dimsum_train --no_filter_candidates --model $BASE_MODEL
python scripts/preprocessing/assign_gold_senses.py data/dimsum_train_train.autoneg.jsonl 
mkdir data/dimsum_train_train
mv data/dimsum_train_train.* data/dimsum_train_train

python scripts/preprocessing/auto_add_negative_mwes.py data/cupt_train_train.jsonl \
--target_percent 1.0 --pipeline cupt_train --no_filter_candidates --model $BASE_MODEL
python scripts/preprocessing/assign_gold_senses.py data/cupt_train_train.autoneg.jsonl 
mkdir data/cupt_train_train
mv data/cupt_train_train.* data/cupt_train_train
 
python scripts/training/train.py \
--data mixed_finetune \
--load_model $BASE_MODEL \
--max_epochs 3 \
--batch_size 16 \
--accumulate_grad_batches 2 \
--gpus 1 \
--swa true \
--gradient_clip_val 1.0 \
--lr 0.00001 \
--run_name mixed_finetune \
--enable_checkpointing true \
--mwe_processing true \
--wsd_processing false \
--limit_val_batches 0 \
--mwe_eval_pipelines cupt_sample dimsum_sample \
--checkpoint_metric val/cupt_sample/mwe_pipeline_f1 \
--train_data_suffix autoneg.sense \
--limit_key_candidates False
```

To finetune on only a single dataset, just change the data argument to `--data cupt` or `--data dimsum`.

## Evaluation

### WSD Eval

```shell
python scripts/training/wsd_eval.py --data data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/ --model $MODEL
```


### MWE Eval
If the dimsum scorer errors out, it may be necessary to comment out lines 204, 206, 456 and 458-472. 
This does not change MWE scoring, but prevents errors from happening. 
```shell
python scripts/training/mwe_out.py cupt_train
python data/parseme_mwe/bin/evaluate.py --pred cupt_train_hyp.cupt --gold data/parseme_mwe/EN/train.cupt 

python scripts/training/mwe_out.py dimsum_train
#  dimsum requires Python 2.7
conda activate py27
python data/dimsum-data/scripts/dimsumeval.py data/dimsum-data/dimsum16.train dimsum_train_hyp.dimsum 

```

### Scores on test data
```shell
python scripts/training/mwe_out.py cupt_test --model $MODEL  --output compare/results.cupt
python scripts/training/mwe_out.py dimsum_test --model $MODEL  --output compare/results.dimsum 

python data/parseme_mwe/bin/evaluate.py --pred compare/results.cupt  --gold data/parseme_mwe/EN/test.cupt 
conda activate py27
python data/dimsum-data/scripts/dimsumeval.py data/dimsum-data/dimsum16.test compare/results.dimsum 


python scripts/training/wsd_eval.py --data data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ --model $MODEL --batch_size 2
```
