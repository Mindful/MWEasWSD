# MWE as WSD
Repo for the paper `MWE as WSD: Solving Multiword Expression Identification with Word
Sense Disambiguation`.

## Installation
```shell
cd MWEasWSD
pip install -e .
```

## Data

```shell
cd data && ./get-data.sh
```

While our repository contains all the code necessary to add
synthetic negative examples to the SemCor data, we also make the fully
processed data with synthetic negatives and our annotations available
in the `data/augmented` directory. Note that the data has been converted to a JSON format. 
If you use the SemCor data in any capacity, please cite the original 
authors as mentioned [here](http://lcl.uniroma1.it/wsdeval/training-data).

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
Pre-trained models can be downloaded from the Hugging Face Hub. 

| Data                    | Architecture     | Download                                        |
|-------------------------|------------------|-------------------------------------------------|
| SemCor                  | Bi-encoder       | https://huggingface.co/Jotanner/mweaswsd        |
| SemCor + PARSEME/DiMSUM | Bi-encoder       | https://huggingface.co/Jotanner/mweaswsd-ft     |
| SemCor                  | DCA Poly-encoder | https://huggingface.co/Jotanner/mweaswsd-dca    |
| SemCor + PARSEME/DiMSUM | DCA Poly-encoder | https://huggingface.co/Jotanner/mweaswsd-dca-ft |

## Training

To replicate our top run Bi-encoder run:
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
--enable_checkpointing true \
--mwe_processing true \
--train_data_suffix fixed.annotated.autoneg
```

For the distinct codes attention Poly-encoder:
```shell
python scripts/training/train.py \
--max_epochs 15 \
--batch_size 16 \
--accumulate_grad_batches 2 \
--gpus 1 \
--swa false \
--gradient_clip_val 1.0 \
--lr 0.00001 \
--run_name replicate-top-dca \
--encoder bert-base-uncased \
--enable_checkpointing true \
--weight_decay 0.01 \
--dropout 0.1 \
--mwe_processing true \
--head_config configs/poly_distinct_codes_128.yaml \
--train_data_suffix fixed.annotated.autoneg
```


### Finetune on DiMSUM/PARSEME data
```shell
export BASE_MODEL=checkpoints/replicate-top/ep14_0.73f1.ckpt # set this to the base model you want to fine tune from

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

```shell
export MODEL=checkpoints/replicate-top/ep14_0.73f1.ckpt # set this to the base model you want to evaluate

python scripts/training/wsd_eval.py --data data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ --model $MODEL --batch_size 2

python scripts/training/mwe_out.py cupt_test --model $MODEL  --output compare/results.cupt
python scripts/training/mwe_out.py dimsum_test --model $MODEL  --output compare/results.dimsum 

python data/parseme_mwe/bin/evaluate.py --pred compare/results.cupt  --gold data/parseme_mwe/EN/test.cupt 
conda activate py27 # dimsum eval requires python 2.7
python data/dimsum-data/scripts/dimsumeval.py data/dimsum-data/dimsum16.test compare/results.dimsum 
```

If the dimsum scorer errors out, it may be necessary to comment out lines 204, 206, 456 and 458-472. 
This does not change MWE scoring, but prevents errors from happening. 
