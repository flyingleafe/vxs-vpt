#!/usr/bin/env bash
echo "Running all the generation routines for the project"

echo "Fetching individual samples from AVP Dataset..."
echo "Fixed subset"
python cut_dataset.py avp ../data/avp-dataset/AVP_Dataset ../data_temp/avp-cut --subset Fixed
echo "Personal subset"
python cut_dataset.py avp ../data/avp-dataset/AVP_Dataset ../data_temp/avp-cut --subset Personal

echo "Fetching individual samples from beatboxset1 (DR annotations)..."
python cut_dataset.py beatboxset1 ../data/beatboxset1 ../data_temp/beatboxset1/DR --anno_type DR

if [ -f "../data/drum_kit_rnn.mag" ]; then
    echo "DrumsRNN model seems to be present"
else
    echo "Downloading pre-trained DrumsRNN model..."
    wget http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag -O ../data/drum_kit_rnn.mag
fi

echo "Generating synthesized tracks using AVP Fixed dataset soundfonts..."
python generate_tracks.py -n 10 --seed 42 ../soundfonts/avp/fixed ../data_temp/avp-gen
echo "Generating synthesized tracks using AVP Personal dataset soundfonts..."
python generate_tracks.py -n 10 --seed 44 ../soundfonts/avp/personal ../data_temp/avp-gen-personal
echo "Generating synthesized tracks using Beatboxset1 soundfonts..."
python generate_tracks.py -n 10 --seed 69 ../soundfonts/beatboxset1 ../data_temp/beatboxset1-gen

if [ -f "../data/200-drum-machines/annotation.csv" ]; then
    echo "200 Drum Machines seems to be partially annotated"
else
    echo "Annotating 200 Drum Machines (partially)"
    python annotate_200drums.py ../data/200-drum-machines
fi
     
