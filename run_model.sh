
# split train set to test and train
python split_data.py

# create features from test files
# TODO: for some reason extract ngrams doesn't work from bash after I downloaded numpy and capstone
for i in {1..9}
do
    python extract_ngrams.py -n $i -p data/train_set.csv
done

python join_ngrams.py

python extract_segments.py

## run model on splitted data
python model.py  # +args