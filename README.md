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

## Fit-scale

For GemNet-T, GemNet-dT, GemNet-GP, GemNet-OC, and PaiNN, the corresponding scale file is __required__.  
For the other models, skip this step.  

The following command generates the scale file.

```
python main.py --mode fit-scale --config-yml $CONFIG \
    --scale-path $SCALEDIR \
    --scale-file $SCALEFILE
```
You should specify `CONFIG`, `SCALEDIR`, and `SCALEFILE`.

After the generated scale file is specified in the model training configuration file, you can train the models.

## Train

```
python main.py --mode train --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID
```
You should specify `CONFIG`, `RUNDIR`, and `RUNID`.

## Train : Resume from a checkpoint
```
python main.py --mode train --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --checkpoint $CKPT_PATH
```
You should specify `CONFIG`, `RUNDIR`, and `CKPT_PATH`.
Checkpoints and logging files will be appended to the same path which was used to train the checkpoint.

## Validate

```
python main.py --mode validate --config-yml $CONFIG \
    --checkpoint $CHECKPOINT \
    --validate-data $VALDATA
```
You should specify `CONFIG`, `CHECKPOINT`, and `VALDATA`.

`CONFIG` which was used for `train` mode can be found in the checkpoint directory.

For now (2023. 03. 24), the available dataset format is only LMDB (`.lmdb`), as in __OCP github__.

If you have a data whose format is `.xyz` or `.extxyz`, please see the section below, where describe an evaluation mode with the energy and force prediction.


## Run MD simulation

```
python main.py --mode run-md \
    --md-config-yml $MD_CONFIG \
    --checkpoint $CHECKPOINT \
    --initial-structure $INIT_STRUCTURE
```
You should specify `MD_CONFIG`, `CHECKPOINT`, and `INIT_STRUCTURE`.

In `MD_CONFIG`, MD simulation conditions such as temperature should be described.

`INIT_STRUCTURE` has `.xyz` format which can be accessed by ASE library.

## Evaluate 

```
python main.py --mode evaluate \
    --evaluation-metric $METRIC \
    --checkpoint $CHECKPOINT \
    --reference-trajectory $VASP_TRAJ \
    --generated-trajectory $MLFF_TRAJ
```
You should specify `METRIC`, `CHECKPOINT`, `VASP_TRAJ`, and `MLFF_TRAJ`.

`METRIC` can be chosen from the list below.  
The two types (the abbreviation and its full name) are available.
* `ef` (or `energy-force`)
* `rdf` (or `radial-distribution-function`)
* `adf` (or `angular-distribution-function`)
* `eos` (or `energy-of-state`)
* `pew` (or `potential-energy-well`)

`VASP_TRAJ` and `MLFF_TRAJ` have `.xyz` format which can be accessed by ASE library.  
`MLFF_TRAJ` is not used in `energy-force` evaluation.


## Deployment for LAMMPS (NequIP, Allegro)

If users utilize a trained model in LAMMPS as a pair, the checkpoint should be deployed according to the guidelines described in NequIP and Allegro github.

To this end, we provide a deploying code, named `deploy_model_for_LAMMPS.py`, that convert a checkpoint trained by our code into the object compatible with LAMMPS.

The following is usage of this code.

```
# for NequIP
python deploy_model_for_LAMMPS.py NequIP $CKPT_OF_NEQUIP

# for Allegro
python deploy_model_for_LAMMPS.py Allegro $CKPT_OF_ALLEGRO
```

If a CKPT is `checkpoint.pt`, then `checkpoint_deploy.pt` will be generated at the same directory.

**Note**: When using the code, please check the training environment and datasets.  In the code, `metadata` (lines 63~74) may be changed accordingly.


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
