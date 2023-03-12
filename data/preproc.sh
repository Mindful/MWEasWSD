# Convert the data into .jsonl files
python scripts/preprocessing/wsd_xml.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor
for d in "data/WSD_Evaluation_Framework/Evaluation_Datasets/se"*"/";
  do python scripts/preprocessing/wsd_xml.py "$d"
done

python scripts/preprocessing/connl_u.py --data_type cupt data/parseme_mwe/EN/train.cupt data/cupt_train.jsonl
python scripts/preprocessing/connl_u.py --data_type cupt data/parseme_mwe/EN/test.cupt data/cupt_test.jsonl

python scripts/preprocessing/connl_u.py --data_type streusle \
data/streusle/test/streusle.ud_test.conllulex \
data/streusle_test.jsonl

python scripts/preprocessing/connl_u.py --data_type streusle \
data/streusle/dev/streusle.ud_dev.conllulex \
data/streusle_dev.jsonl

python scripts/preprocessing/dimsum.py data/dimsum-data/dimsum16.test data/dimsum_test.jsonl
python scripts/preprocessing/dimsum.py data/dimsum-data/dimsum16.train data/dimsum_train.jsonl

# Generate data using scripts/preprocessing/PrintData.java if necessary (see comments in file for instructions)
python scripts/preprocessing/kulkarni.py data/kulkarni_finlayson_data.txt

# Samples
python scripts/preprocessing/random_sample.py data/kulkarni.jsonl
python scripts/preprocessing/random_split.py data/cupt_train.jsonl
python scripts/preprocessing/random_split.py data/dimsum_train.jsonl
python scripts/preprocessing/fix_existing_mwes.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.jsonl