# Train Script (`--mode train`)

## Command 1: Train from scratch

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
* `--save-ckpt-every-epoch N` (optional) : save a model parameters checkpoint at every N epoch (default: None, meaning that the intermediate heckpoints are not saved during training)


### Convenient Training Script

```
./run_train.sh $GPU $MODEL $DATA

# example:
# ./run_train.sh 0 GemNet-dT SiN
```

This means that GemNet-dT is trained using the SiN dataset and 0th GPU (with [the configuration file](../../configs/train/SiN/GemNet-dT.yml))

In this script, options used in our benchmark experiments, for `MODEL` and `DATA`, are listed.


### Modify Training Hyperparameters

To modify training hyperparameters, please refer [the directory](../../configs/) that includes training configurations of various models for SiN and HfO.

*Note* :  In our benchmark paper, two loss functions are explained: MSE- and MAE-based losses.  These two losses can be seen in each configuration file. Just **activate only one loss function**.

## Command 2: `Resume` Training of a Checkpoint
```
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --checkpoint $CKPT_PATH \
    --resume \
```
Users should specify `CONFIG`, `RUNDIR`, and `CKPT_PATH`.

Resuming is to resume the training of the given checkpoint. 
If a training process is accidently stopped, the resuming functional can be helpful.
The checkpoint should include the training states such as optimizers; that is, users just use `checkpoint.pt` saved at the checkpoint directory.
When resuming, checkpoints and logging files will be appended to the same path which was used to train the checkpoint.

When running the command 1 (training from scratch), the configuration file used for the training is saved at the checkpoint directory. 
Hence, users conveniently use the saved configuration file as `CONFIG`.

The arguments mean as follows:
* `--checkpoint` (optional) : a checkpoint path to used for resuming the training of checkpoint (default: None)
* `--resume` (optional) : determine to resume the training of the given checkpoint (default: False, meaning that finetuning the model from the checkpoint)


## Command 3: `Finetune` the Model from a Checkpoint
```
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID \
    --checkpoint $CKPT_PATH \
```
Users should specify `CONFIG`, `RUNDIR`, `RUNID`, and `CKPT_PATH`.

Finetuning is to finetune a model whose initial weights are set by the given checkpoint.
Unlike resuming, it is OK that the checkpoint just includes model weights.
The training options are set by the configuration options specified in `CONFIG`.

The arguments mean as follows: 
* `--checkpoint` (optional) : a checkpoint path to used for finetuning a model from the checkpoint (default: None)