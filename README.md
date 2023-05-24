# AL-foundation-models
Active Learning in the era of Foundation Models

### Installation instructions
```
conda env create -f env.yml
conda activate alfm_env

git clone git@github.com:tempconfx/AL-foundation-models.git
cd Al-foundation-models
conda develop .
```

Make a copy of `ALFM/.env.example` and name it `ALFM/.env`. Change the paths in it to reflect your local setup

### Feature extraction
Feature extraction sweeps are configured with `ALFM/conf/feature_extraction.yaml`.

Run `python -m ALFM.feature_extraction` to run the sweep. 

Specify command line overrides with `python -m ALFM.feature_extraction model=dino_vit_S14 dataset=cifar10`.

Features will be saved to `FEATURE_CACHE_DIR` as specified in your `.env` file.

For multi-GPU inference, make the following changes
```
export SLURM_JOB_NAME=interactive  # otherwise pytorch lightning won't launch new processes
python -m ALFM.feature_extraction +trainer.strategy=ddp, trainer.devices=4
```

### Active Learning experiments

Run `python -m ALFM.al_train` to run an Active Learning experiment. 

Default values are set in `ALFM/conf/al_training.yaml` but can be overriden.

Specify command line overrides with `python -m ALFM.al_train model=dino_vit_g14`

Guaranteed to be super fast ;)