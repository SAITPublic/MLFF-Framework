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
* `--print-every N` (optional) : print a log at every N step (default: 10)
* `--save-ckpt-every-epoch N` (optional) : save a model parameters checkpoint at every N epoch (default: None, meaning that cthe intermediate heckpoints are not saved during training)


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

## Additional Command : `Finetune` the Model from a Checkpoint or `Resume` Training of a Checkpoint
```
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --checkpoint $CKPT_PATH \
    --resume \
```
Users should specify `CONFIG`, `RUNDIR`, and `CKPT_PATH`.

When resuming, checkpoints and logging files will be appended to the same path which was used to train the checkpoint.

The arguments mean as follows:
* `--checkpoint` (optional) : a checkpoint path to used for resuming the training of checkpoint or finetuning a model from the checkpoint (default: None)
* `--resume` (optional) : determine to resume the training of the given checkpoint (default: False, meaning that finetuning the model from the checkpoint)