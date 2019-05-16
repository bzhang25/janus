[![Build Status](https://travis-ci.org/CCQC/janus.svg?branch=master)](https://travis-ci.org/CCQC/janus)
[![codecov](https://codecov.io/gh/CCQC/janus/branch/master/graph/badge.svg)](https://codecov.io/gh/CCQC/janus)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# <img src="https://github.com/bzhang25/janus/blob/master/images/janus_logo_final.png" width="150">
A Python library for adaptive QM/MM methods <br/>
Documentation can be found [here](https://ccqc.github.io/janus/)

## Set up steps: 
* Download miniconda [installer](https://conda.io/docs/user-guide/install/macos.html)
* Go to folder where this is downloaded and run: `bash Miniconda3-latest-OSX-x86_64.sh -b -p $HOME/miniconda`
* To set conda path to miniconda: `echo PATH="\$HOME/miniconda/bin:\$PATH" >> ~/.bash_profile`
* Create environment in miniconda: `conda create -n janus python=3.6 psi4 psi4-rt numpy openmm -c psi4 -c omnia`
* To activate the environment: `source activate janus`
* To install locally: `pip install -e .`
* For reading external datafiles with pytest within the janus environment: `pip install pytest-datafiles` 
* Adding mendeleev package: `pip install mendeleev` 
* To add MDtraj: `conda install -c conda-forge mdtraj`
* To add sphinx : `pip install Sphinx`
* To add napoleon extension to sphinx `pip install sphinxcontrib-napoleon`


