# SAIT-MLFF-Framework

We provide four functionals named `fit-scale`, `train`, `validate`, `run-md`, and `evaluate`.

The explanation about how to operate these functionals is described as below.

For using more arguments of these functionals, it would be helpful to see `scripts/` and `configs/`.

## Preparation

After uncompressing `codes.zip` and following the instructions below, users can install the framework and perform MLFF benchmarks.

### Install

```
cd SAIT-MLFF-Framework
```

From now on, the base working directory is the inside of `SAIT-MLFF-Framework/`.

By the following instructions, the packages related to MLFF models and MD simulation are downloaded (git clone).

```
git submodule init
git submodule update
```

We modify [OCP](https://github.com/Open-Catalyst-Project/ocp) and [auto-FOX](https://github.com/nlesc-nano/auto-FOX), which are located in `codebases/`, with minor modifications.  
To enable users apply the modifications, we provide [two patch files](codebases/patches/).  
The following instructions perform applying the patch to each submodule.

```
# auto-FOX
cd codebases/auto-FOX
git apply ../patches/auto-FOX-custom.patch
pip install .

# OCP
cd ../ocp
git apply ../patches/ocp-scn-custom.patch
```

Users do not need to explicitly install MLFF packages required by SAIT-MLFF-Framework, such as OCP, NequIP, and etc., using `pip`.  

*Note* : Any other MLFF package can be compatible with our framework if some requirements are satified as follows.
* The wrapper for models supported by the package should be implemented (see `src/common/models`).
* If the package is located at `codebases/`, `sys.path` should include its path (see `main.py`).
* If data format used by models is different from that of [OCP](https://github.com/Open-Catalyst-Project/ocp), data that is loaded from `.lmdb` (prepared by our script) should be converted into the data format of the package (see `src/common/collaters/`).
* If some training conditions need to be handled, a tailored trainer class should be implemented (see `src/common/trainers/`)


### Download Datasets

Our semiconductor datasets (SiN and HfO) can be downloaded from the following links.
* [SiN (raw)](https://drive.google.com/file/d/1umhok3RbYyjjnpeKkxEGJUN2oY3OxSBN/view?usp=sharing)
* [SiN](https://drive.google.com/file/d/1l9nsie40Bpm8CNW4sx94yAuvmMkUfM3b/view?usp=sharing)
* [HfO (raw)](https://drive.google.com/file/d/1tSkjfp4N8cvHqpFYYlu2EqK8u2HRIro7/view?usp=sharing)
* [HfO](https://drive.google.com/file/d/1-DVMGyXjvNYaBtaAkWu8uQVgvz8pEgMZ/view?usp=sharing)

```
# extract tar files at the datasets directory
cd datasets
tar xf SiN.tar
tar xf HfO.tar

# optional
rm SiN.tar
rm HfO.tar
```

### Data Preprocesing 

The preprocessing, which converts the .xyz into .lmdb, is explained in [this](scripts/preprocess_data/).

## Train MLFF Models

### Fit-scale

The details are explained in [this](scripts/fit_model_scale_factors/).

### Train

The details are explained in [this](scripts/train/).

### Validate

The details are explained in [this](scripts/validate/).

For now, the available dataset format is only LMDB (`.lmdb`), as in [OCP](https://github.com/Open-Catalyst-Project/ocp).  
If users have a data whose format is `.xyz` or `.extxyz` and want to check errors of energy and forces without the data preprocessing, please refer the evaluation mode for energy and force prediction below.

## Evaluation

### Evaluate using Errors of Energy and Force

The details are explained in [this](scripts/evaluate/README.md#evaluate-errors-of-energy-and-forces).

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
