#!/bin/bash

if [ ! -d ".env" ]; then
	virtualenv -p python3 .env
fi
. .env/bin/activate

apt-get install cython python-scipy python-pip python-lxml python-dev

# upgrade pip
pip install --upgrade pip
# install common libraries
pip install scipy lxml
