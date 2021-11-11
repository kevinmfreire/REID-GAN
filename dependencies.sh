#!/bin/bash

set -e
sudo apt install python3-pip
pip install -f https://download.pytorch.org/whl/torch_stable.html torch===1.4.0+cu101
pip install -r requirements.txt