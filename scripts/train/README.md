# Train Script

## Command (`--mode train`)

```
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID \
    --print-every N \
    --save-ckpt-every-epoch N \
```
Users should specify `CONFIG`, `RUNDIR`, and `RUNID`.

The arguments mean as follows:

* `--config-yml` : a configuration file to train an MLFF model
* `--run-dir` : a path to save training results
* `--identifier` : an identifier of a training trial, where users can describe the training.
* `--print-every N` : print a log at every N step (iteration)
* `--save-ckpt-every-epoch N` : save a model parameters checkpoint at every N epoch


## Convenient Training Script

```
./run_train.sh $GPU $MODEL $DATA

# example:
# ./run_train.sh 0 GemNet-dT SiN
```

This means that GemNet-dT is trained using the SiN dataset and 0th GPU (with [the configuration file](../../configs/train/SiN/GemNet-dT.yml))

In this script, options used in our benchmark experiments, for `MODEL` and `DATA`, are listed.


## Modify Training Hyperparameters

To modify training hyperparameters, please refer [the directory](../../configs/) that includes training configurations of various models for SiN and HfO.

*Note* :  In our benchmark paper, two loss functions are explained: MSE- and MAE-based losses.  These two losses can be seen in each configuration file.   Just **activate only one loss function**.

## Resume Training from a Checkpoint
```
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --checkpoint $CKPT_PATH
```
Users should specify `CONFIG`, `RUNDIR`, and `CKPT_PATH`.
Checkpoints and logging files will be appended to the same path which was used to train the checkpoint.