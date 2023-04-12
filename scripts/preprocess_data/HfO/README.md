# HfO (SAIT)

## Download

URL is not available.

In ESC project (SAIT), the data path is /mnt/DB/HfO_v1.0/

## Dataset

There are four dataset generated from VASP simulation for HfO.

### In-sample 1
MD simulations whose starting structures are 5 __crystalline__ structures.  
Train/valid/test splitting is 8:1:1.


### In-sample 2
MD simulations whose starting structures are 5 __random__ structures.  
Train/valid/test splitting is 8:1:1.


### In-sample 3
Dataset which includes 'In-sample 1' and 'In-sample 2'.  
Train/valid/test splitting is 8:1:1.


### OOS: Out-of-Sample (OOD: Out-of-Distribution)

MD simulations whose starting structure is a __ramdom__ structure which is not observed in 'In-sample 1' and 'In-sample 2'.


## Split (train/valid/test)

The split for each dataset is prepared by Seung jin Kang (sj1222.kang).
The details will be filled.

(or the split option in preprocess.py can be implemented by Seung jin Kang)

### Split 1



## Construct graphs using a cutoff radius
If a cutoff radius is given, atom cloud data can be converted into graph data and saved with edge indices.  