# Validate Script

## Command (`--mode validate`)

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode validate \
    --config-yml $CONFIG \
    --checkpoint $CKPT_PATH \
    --validate-data $DATA_LMDB \
    --validate-batch-size 16 \
```
You should specify `CONFIG`, `CKPT_PATH`, and `DATA_LMDB`.

The arguments mean as follows:

* `--config-yml` : a configuration file that was used to train the checkpoint (which was automatically saved at the checkpoints directory of this training result)
* `--checkpoint` : a checkpoint of an MLFF model
* `--validate-data` : a data file which should be __.lmdb__
* `--validate-batch-size N` : using a batch size of N (if not given, it is set as a batch size in the configuration file)


## Convenient Validation Script

```
./run_validate.sh $GPU $CKPT_DIR $DATA_LMDB

# example:
# ./run_validate.sh 0 ../../train_results/SiN/GemNet-dT/checkpoints/train-20230101_010000 ../../datasets/SiN/ood/atom_cloud/ood.lmdb
```
In this script, the checkpoint used for validation is the last checkpoint.

