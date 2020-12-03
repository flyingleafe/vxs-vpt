# vxs-vpt

Additional material for MSc dissertation "Deep Feature Extraction and Music Language Modelling for Amateur Vocal Percussion Transcription"

## Environment setup

Install prerequisites for fluidsynth

```
apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev
```

Install python libraries (preferrably inside a virtualenv):
```
pip install -r requirements.txt
```

Download all the data (caution, significant data sizes!):
```
./fetch_data.sh
```

Download DrumsRNN model and do data preprocessing (samples cutting, synthesis, etc.):
```
cd scripts
./run_all.sh
```
One of the sounds in 200 drums machines dataset is corrupted and should be deleted manually. You will see which in an IO error message.

## Model training
Train CAEs (final version, with barkgrams)
```
cd scripts
python cae_training.py ../configs/cae_training_bark.yaml
```
Train DNN classifiers (require CAEs to be trained first):
```
cd scripts
python classifier_training.py ../configs/classifier_training_2.yaml
```

## Experiments
Onset detection results are in `notebooks/Onsets.ipynb`

Main results are in `notebooks/Classification.ipynb`

LM influence analysis is in `notebooks/LMAnalysis.ipynb`

Replication of Mehrabi's results is split between `LMER.ipynb` and `LMER-R.ipynb`, the second requires R and R kernel for Jupyter to be installed.
