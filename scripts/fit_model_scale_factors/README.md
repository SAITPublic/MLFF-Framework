# Fit-scale Script

For each GemNet-T and GemNet-dT, the corresponding scale file is __required__.  
For the other models, skip this step.  

*Note* : When generating a scale file, the configuration file of GemNet-T or GemNet-dT **should not include the argument named `scale_file`.**

After the generated scale file is specified in the model training configuration file, users can train the models.  
The scale files used in our benchmark are provided in [SiN scale files](configs/train/SiN/auxiliary/) and [HfO scale files](configs/train/HfO/auxiliary/).


## Command (`--mode fit-scale`)

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode fit-scale \
    --config-yml $CONFIG \
    --scale-path $SCALE_DIR \
    --scale-file $SCALE_FILE \
```
You should specify `CONFIG`, `SCALE_DIR`, and `SCALE_FILE`.

The arguments mean as follows:

* `--config-yml` : a configuration file that was used to train the checkpoint (which was automatically saved at the checkpoints directory of this training result)
* `--scale-path` : a directory path to save the generated scale file
* `--scale-file` : a filename of the scale file


## Convenient Fit-scale Script

```
./run_fit_scale.sh $GPU $MODEL $DATA

# example:
# ./run_validate.sh 0 ../../train_results/SiN/GemNet-dT/checkpoints/train-20230101_010000 ../../datasets/SiN/ood/atom_cloud/ood.lmdb
```
In this script, the checkpoint used for validation is the last checkpoint.

