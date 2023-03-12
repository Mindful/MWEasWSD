python compute_mwe_vocab.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor mwe_vocab.txt
python preproc_add_mwe_data.py data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.jsonl --add_mwes --mwe_vocab mwe_vocab.txt
head data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.added_mwe.jsonl -n 18588 > semcor_mwe_first_half.jsonl
tail data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.added_mwe.jsonl -n 18588 > semcor_mwe_second_half.jsonl