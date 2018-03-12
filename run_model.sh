
# split train set to test and train
python split_data.py data/train_labels_filtered.csv

# create features from test files
for i in {1..9}
do
    python extract_ngrams.py data/files -n $i -p data/train_set.csv
done

if [ ! -d "features" ]; then
    mkdir features
fi

# decide on final features
python join_ngrams.py
python extract_segments.py

# create f2v files
if [ ! -d "f2v" ]; then
    mkdir features
fi
python f2v.py data/files data/train_set.csv f2v/train.f2v
python f2v.py data/files data/test_set.csv f2v/test.f2v

## run model on splitted data
python model.py  -train -create-f2v data/files data/train_labels_filtered.csv f2v.file -save first.model
