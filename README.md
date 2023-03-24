# SAIT-MLFF-Framework

We provide four functionals named `train`, `fit-scale`, `validate`, and `evaluate`.

The explanation about how to operate these functionals is described as below.

For using more arguments of these functionals, it would be helpful to see `scripts/` and `NeurIPS23_DnB/`.

## Install

```
git clone --recurse-submodules https://github.sec.samsung.net/ESC-MLFF/SAIT-MLFF-Framework.git

cd SAIT-MLFF-Framework
```

Your base working directory is `SAIT-MLFF-Framework/`.

You do not need to install packages required by SAIT-MLFF-Framework, such as OCP, NequIP, and etc., using `pip`.

## Train

```
python main.py --mode train --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID
```
You should specify `CONFIG`, `RUNDIR`, and `RUNID`.

## Fit-scale

For GemNet-T, GemNet-dT, GemNet-GP, GemNet-OC, and PaiNN, the corresponding scale file is required.
The following command generates the scale file.

```
python main.py --mode fit-scale --config-yml $CONFIG \
    --scale-path $SCALEDIR \
    --scale-file $SCALEFILE
```
You should specify `CONFIG`, `SCALEDIR`, and `SCALEFILE`.

After the generated scale file is specified in the model training configuration file, you can train the models.


## Validate

```
python main.py --mode validate --config-yml $CONFIG \
    --checkpoint $CHECKPOINT \
    --validate-data $VALDATA
```
You should specify `CONFIG`, `CHECKPOINT`, and `VALDATA`.

`CONFIG` which was used for `train` mode can be found in the checkpoint directory.

For now (2023. 03. 24), the available dataset format is only LMDB (.lmdb), as in __OCP github__.


## Evaluate

To be implemented.

## Acknowledge and Reference Code
SchNet paper  
DimeNet and DimeNet++ paper  
GemNet paper  
GemNet-OC paper  
NequIP paper  
Allegro paper  
MACE paper  

OCP github  
NequIP github  
Allegro github  
MACE github  
SIMPLE-NN github  
