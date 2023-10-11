# Convert Script (for NequIP and Allegro)

NequIP and Allegro models, which were trained without stress, can be converted into models that can predict stress, and *vice versa*.

## Command

```
# convert the model of force prediction into the model of force and stress prediction
python convert_force_to_stress.py \
    --ckpt-dir $CKPT_DIR \
    --out-dir $OUT_DIR \
    --ckpt-name $CKPT_NAME \

# convert the model of force and stress prediction into the model of force prediction
python convert_stress_to_force.py \
    --ckpt-dir $CKPT_DIR \
    --out-dir $OUT_DIR \
    --ckpt-name $CKPT_NAME \
```
You should specify `CKPT_DIR` and `OUT_DIR`.

The arguments mean as follows:

* `--ckpt-dir` : a directory that includes a checkpoint file and a configuration file used for training
* `--out-dir` : a directory to save the converted results
* `--ckpt-name` (optional) : a checkpoint filename which is located at `CKPT_DIR` (default: checkpoint.pt)

