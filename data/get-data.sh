# run this from bunmyaku/data
# https://sapienzanlp.github.io/xl-wsd/
#gdown https://drive.google.com/uc?id=19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0
# the above link doesn't work, I grabbed below link for in-browser download button
wget "https://drive.google.com/u/0/uc?id=19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0&confirm=t&uuid=f7343ad2-1152-42c0-918f-305bdfb39db1&at=ALgDtsyvqvsTWOC5BmrQvN8uSugP:1675620446367" -O xl-wsd-data.zip
unzip -q xl-wsd-data.zip

# http://lcl.uniroma1.it/wsdeval/
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip -q WSD_Evaluation_Framework.zip

rm *.zip

# https://gitlab.com/parseme/corpora/-/wikis/home#languages
mkdir parseme_mwe && cd parseme_mwe
wget "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-2842/EN.tgz"
tar zxvf EN.tgz
wget "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-2842/bin.tgz"
tar zxvf bin.tgz
rm *.tgz

wget http://projects.csail.mit.edu/jmwe/download.php?f=mweindex_wordnet3.0_semcor1.6.data -O mweindex_wordnet3.0_semcor1.6.data

cd ..

#python -c "import nltk;nltk.download('wordnet')"
#python -c "import nltk;nltk.download('omw')"
#python -c "import nltk;nltk.download('omw-1.4')"
#python -c "import nltk;nltk.download('words')"
wget https://wordnetcode.princeton.edu/1.6/wn16.unix.tar.gz
tar -xf wn16.unix.tar.gz
mkdir wordnet-1.6/wordnet
cp wordnet-1.6/dict/index.sense wordnet-1.6/wordnet/
mv wordnet-1.6/ ~/nltk_data/
python -m spacy download en_core_web_sm

git clone git@github.com:dimsum16/dimsum-data.git
#git clone git@github.com:nert-nlp/streusle.git


gdown https://drive.google.com/uc?id=1Vb0uRDxVsvPIgJhBctbEwyFBvcSZOzR5  # kulkarni.jsonl
gdown https://drive.google.com/uc?id=1-i4AH65Zkh41onQJjKGWgQ9c2tPH8NLP  # kulkarni_finlayson_data.txt