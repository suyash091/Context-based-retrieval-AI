#!/usr/bin/env bash

# download punkt first
python2 download_punkt.py

python2 create_ubuntu_dataset.py "$@" --output 'train.csv' 'train'
python2 create_ubuntu_dataset.py "$@" --output 'test.csv' 'test'
python2 create_ubuntu_dataset.py "$@" --output 'valid.csv' 'valid'
