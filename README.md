# SAIT-MLFF-Framework

We provide four functionals named `fit-scale`, `train`, `validate`, `run-md`, and `evaluate`.

The explanation about how to operate these functionals is described as below.

For using more arguments of these functionals, it would be helpful to see `scripts/` and `configs/`.

## Preparation

### Install (수정 필요 -> AutoFox)

```
git clone --recurse-submodules https://github.sec.samsung.net/ESC-MLFF/SAIT-MLFF-Framework.git

cd SAIT-MLFF-Framework
```

From now on, the base working directory is the inside of `SAIT-MLFF-Framework/`.

Users do not need to explicitly install MLFF packages required by SAIT-MLFF-Framework, such as OCP, NequIP, and etc., using `pip`.  
They exist in `codebases/` by git-cloning.

*Note* : Any other MLFF package can be compatible with our framework if some requirements are satified as follows.
* The wrapper for models supported by the package should be implemented (see `src/common/models`).
* If the package is located at `codebases/`, `sys.path` should include its path (see `main.py`).
* If data format used by models is different from that of [OCP](https://github.com/Open-Catalyst-Project/ocp), data that is loaded from `.lmdb` (prepared by our script) should be converted into the data format of the package (see `src/common/collaters/`).
* If some training conditions need to be handled, a tailored trainer class should be implemented (see `src/common/trainers/`)


### Download Datasets (데이터 압축 포맷, 링크 필요)

* SiN [link]()
* HfO [link]()

```
mkidr datasets
wget [link] data.zip
unzip data.zip
mv data/SiN datasets/SiN
mv data/HfO datasets/HfO
mv data/SiN_raw datasets/SiN_raw
mv data/HfO_raw datasets/HfO_raw
rm -rf data/
```

### Data Preprocesing 

The preprocessing, which converts the .xyz into .lmdb, is explained in [this](scripts/preprocess_data/).

## Train MLFF Models

### Fit-scale

For each GemNet-T and GemNet-dT, the corresponding scale file is __required__.  
For the other models, skip this step.  

The details are explained in [this](scripts/fit_model_scale_factors/).

After the generated scale file is specified in the model training configuration file, users can train the models.  
The scale files used in our benchmark are provided in [SiN scale files](configs/train/SiN/auxiliary/) and [HfO scale files](configs/train/HfO/auxiliary/).


### Train

The details are explained in [this](scripts/train/).

### Validate

The details are explained in [this](scripts/validate/).

For now, the available dataset format is only LMDB (`.lmdb`), as in [OCP](https://github.com/Open-Catalyst-Project/ocp).  
If users have a data whose format is `.xyz` or `.extxyz` and want to check errors of energy and forces without the data preprocessing, please refer the evaluation mode for energy and force prediction below.

## Evaluation

### Evaluate using Errors of Energy and Force

The details are explained in [this](scripts/evaluate/README.md#link_energy_force_error).

### Run MD simulation

The details are explained in [this](scripts/simulate/).

### Evaluate using Simulation Indicators

The details are explained in [this](scripts/evaluate/).

## Acknowledge and Reference Code
OCP [github](https://github.com/Open-Catalyst-Project/ocp)   
NequIP [github](https://github.com/mir-group/nequip)   
Allegro [github](https://github.com/mir-group/allegro)  
MACE [github](https://github.com/ACEsuit/mace)  
SIMPLE-NN [github](https://github.com/MDIL-SNU/SIMPLE-NN_v2)   
