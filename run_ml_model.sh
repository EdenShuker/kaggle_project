#!/usr/bin/env bash

# split train set to test and train
python split_data.py data/train_labels_filtered.csv data 0.33

# create features from test files
if [ ! -d "ml_code/ngrams" ]; then
    mkdir ml_code/ngrams
fi
for i in {1..9}
do
    python ml_code/extract_ngrams.py data/files -n $i -p data/train_set.csv
done

if [ ! -d "ml_code/features" ]; then
    mkdir ml_code/features
fi

# decide on final features
python ml_code/join_ngrams.py
python ml_code/extract_segments.py

# create f2v files
if [ ! -d "ml_code/f2v" ]; then
    mkdir ml_code/f2v
fi
python ml_code/f2v.py data/files data/train_set.csv ml_code/f2v/train.f2v
python ml_code/f2v.py data/files data/test_set.csv ml_code/f2v/test.f2v

# run model on splitted data
python ml_code/model.py  -train data/train_set.csv ml_code/f2v/train.f2v -save ml_code/first.model -test data/test_set.csv ml_code/f2v/test.f2v ml_code/test.output
python ml_code/eval_model.py data/test_set.csv ml_code/test.output -show-matrix

#python ml_code/model.py -load ml_code/first_9677.model -test data/train_set.csv ml_code/f2v/train.f2v ml_code/train.output
#python ml_code/eval_model.py data/train_set.csv ml_code/train.output -show-matrix