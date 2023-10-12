# Deploy Script (for NequIP and Allegro)

NequIP and Allegro models can be deployed into pair model used in LAMMPS (feat: [Harvard MIR group](https://github.com/mir-group)).

If users utilize a trained model in LAMMPS as a pair, the checkpoint should be deployed according to the guidelines described in NequIP and Allegro github.

To this end, we provide a deploying code, named `deploy.py`, that convert a checkpoint trained by our code into the object compatible with LAMMPS.

The following is usage of this code.

```
# deploy a NequIP model
python deploy.py NequIP $CKPT_PATH


# deploy an Allegro model
python deploy.py Allegro $CKPT_PATH
```

You should specify `CKPT_PATH`. If `CKPT_PATH` is `checkpoint.pt`, then `checkpoint_deploy.pt` will be generated at the same directory.

*Note 1*: When using the deploying code, please check the training environment and datasets. In the code, `metadata` assignment (lines 74~96) may be changed accordingly.

*Note 2* : To be compatible with LAMMPS, the models should be trained __without the max_neighbors restriction__.



