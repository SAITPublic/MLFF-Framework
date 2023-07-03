# Dataset Preprocessing

## Download

URL for downloding SiN and HfO will be available.

## Convenient Preprocessing Script

To train models using this framework, users should preprare a database (with .lmdb format).

Using [this script](./run_preprocess_data.sh) as following, users can easily do.

```
./run_preprocess_data.py $DATA $OUTDATA_TYPE
```
You should specify `DATA` and `OUTDATA_TYPE`.


### ■ Option 1 (`cloud`)
: save coordinates
```
./run_preprocess_data.py SiN cloud
./run_preprocess_data.py HfO cloud
```

### ■ Option 2 (`graph`)
: save coordinates and edges (which are generated with a cutoff radius of 6.0 and a max number of neighborhood atoms of 50)
```
./run_preprocess_data.py SiN graph 6.0 50
./run_preprocess_data.py HfO graph 6.0 50
```

Some models can generate graphs from coordinates of atoms on-the-fly, but some cannot (such as NequIP, Allegro, and MACE).

For latter models, users should generate graphs in advance and save the graphs into .lmdb file.

If a cutoff radius is given, atom cloud data can be converted into graph data and saved with edge indices.  


## Split Information (train/valid/test)

A dataset is divided into train, validation, and test sets with a ratio of 8:1:1.

The details are described in our benchmark paper.