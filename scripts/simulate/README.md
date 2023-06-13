# Simulation Script

**Before obtaining radial distribution function (RDF) and angular distribution function (ADF), users should run the MD simulation using a target MLFF model**.  
The simulation generates a trajectory which is used to obtain RDF and ADF.  
The other simulation indicators (bulk modulus, equilibrium volume, and potential energy curves) does not need the simulation results.

## Command (`--mode run-md`)

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode run-md \
    --md-config-yml $MD_CONFIG \
    --checkpoint $CKPT_PATH \
```
You should specify `MD_CONFIG` and `CKPT_PATH`.

The arguments mean as follows:

* `--md-config-yml` : a configuration file for MD simulations (including initial structure data with `.xyz` format, temperature, and so on)
* `--checkpoint` : a checkpoint of an MLFF model


## Convenient Simulation Script

There are two scripts for SiN and HfO, respectively.  
The followings are an example for SiN (which is identical to HfO)

```
./run_md_sim_SiN.sh $GPU $MODEL $CKPT_PATH

# example:
# ./run_validate.sh 0 GemNet-dT ../../train_results/SiN/GemNet-dT/checkpoints/train-20230101_010000/ckpt_ep200.pt
```

There is [a template MD configuration file](../../configs/simulate/md_sim_config.yml).

Some variables in the template are replaced by information listed in this script.


## Information

1. In these scripts, 4 SiN and 13 HfO initial structures, whose details are described in our paper (appendix).

2. Using this script, MD simulations using super-cells (2x2x2 and 3x3x3) are automatically executed.

## Running Own Simulation

To run own simulation, users can implement their simulation using calculators with their MLFF models.  
The calculators are prepared in [this directory](../../src/md_evaluate/).  
It would be helpful to see our evaluator classes using the calculators.
