# Validate Script

## Command (`--mode validate`)

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode validate \
    --config-yml $CONFIG \
    --checkpoint $CKPT_PATH \
    --validate-data $DATA_SRC \
    --validate-batch-size 16 \
    --separate-evaluation \
    --shuffle \
```
You should specify `CONFIG`, `CKPT_PATH`, and `DATA_SRC`.

The arguments mean as follows:

* `--config-yml` : a configuration file that was used to train the checkpoint (which was automatically saved at the checkpoints directory of this training result)
* `--checkpoint` : a checkpoint of an MLFF model
* `--validate-data` : a data source path, which can be a single __.lmdb__ file or a directory including multiple __.lmdb__ files
* `--validate-batch-size N` (optional) : a batch size of N used for validation (if not given, it is set as a batch size in the configuration file)
* `--separate-evaluation` (optional) : print the results of multiple data files separately (default: False)
* `--shuffle` (optional) : shuffle the data (default: False)


## Convenient Validation Script

```
./run_validate.sh $GPU $CKPT_DIR $DATA_LMDB

# example:
# ./run_validate.sh 0 ../../train_results/SiN/GemNet-dT/checkpoints/train-20230101_010000 ../../datasets/SiN/ood/atom_cloud/ood.lmdb
```
In this script, the checkpoint used for validation is the last checkpoint.

