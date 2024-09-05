# run this from bunmyaku/data

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

# cd ..

python -m spacy download en_core_web_sm

git clone https://github.com/dimsum16/dimsum-data.git
