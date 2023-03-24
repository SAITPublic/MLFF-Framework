# rMD17

## Download

The revised MD17 (rMD17) can be downloaded from https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038

The filename is rmd17.tar.bz2
> rMD17 data details are as follows.

>> _nuclear_charges_ : The nuclear charges for the molecule  
>> _coords_ : The coordinates for each conformation (in units of ångstrom)  
>> _energies_ : The total energy of each conformation (in units of kcal/mol)  
>> _forces_ : The cartesian forces of each conformation (in units of kcal/mol/ångstrom)


## Split (train/valid/test)

### Default sampling (uniform sampling)
The split is train (950 snapshots), val (50 snapshots), and test (the remaining snapshots), where the data are randomly sampled.
```
python preprocess.py --data-dir RMD17_DATA/npz_data/ --out-path RMD17_DATA_OUTPATH/
```

### Specify the split sizes
You can give the sizes of train and valid dataset, and accordingly the test dataset is the remaining snapshots.
```
python preprocess.py --data-dir RMD17_DATA/npz_data/ --out-path RMD17_DATA_OUTPATH/ --train-size 9500 --val-size 500
```

Also, you can specify only the size of testset.
```
python preprocess.py --data-dir RMD17_DATA/npz_data/ --out-path RMD17_DATA_OUTPATH/ --test-size 1000
```

### Use a sampling method with a fixed step
You can sample train and valid dataset at every given step.
```
python preprocess.py --data-dir RMD17_DATA/npz_data/ --out-path RMD17_DATA_OUTPATH/ --sampling-step 50
```


## Construct graphs using a cutoff radius
If a cutoff radius is given, atom cloud data can be converted into graph data and saved with edge indices.  
> **_Note_ : It is recommend to not pre-compute the graph for rMD17.**  
> The molecules includes small number of atoms, so constructing graph data on-the-fly ('otf_graph: True' in a configuration yaml file) has ignorable overhead.
```
python preprocess.py --data-dir RMD17_DATA/npz_data/ --out-path RMD17_DATA_OUTPATH/ --r-max 5.0
```