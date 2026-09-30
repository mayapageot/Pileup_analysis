# Pileup_Analysis
Software for pileup discrimination analysis

This software is designed to discriminate pileup event from Light detector of the cupid experiment based on CROSS 
experimental data.

It is based on a PSD parameter Y defined as the ratio of 2 energy estimators which response is in turn defined by 2 
different frequency weighting function m1(w) and m2(w). The goal of the software is to optimize the frequency weighting 
functions m_1(w) and m2(w) in order to maximize the discrimination power of the PSD parameter Y. For that it uses an 
analitical computation of the response of the PSD to pileup and compute a pileup misidentification rate M with respect 
to the dt (time between the 2 events), r the relative amplitude of the 2 events and the total energy.

Then it computes the resulting performance in term of Background Index (BI) using noise data from the CROSS experiment 
and injecting selected pulses given previously generated amplitudes and delays.

The work is presented in the following paper: https://arxiv.org/abs/2606.17948

## Tutorial
A good starting point to understand the software is to go through the `tutorial.ipynb` jupyter notebook in the folder
`test` which will guide you through the main steps of the analysis and show you how to use the software.

It is possible to run it without any additional data until the `# Compute the Background Index using injected pulses`
step, which requires the user to provide the noise data (`file_path`) and the montecarto generated amplitude and time 
delay (`path_pos`). This data is available upon request to the author of the paper or on feynman 
(setting DATA_PATH=/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected in the `.env` file) and cupid_login 
(setting DATA_PATH = /data2/LSC/DATA/RUN9 in the `.env` file)

Creating a new dataset requires to select a run binary file of the analysed detector and modify the pos file so that
the position corresponds to noise traces in the binary file.


## Installation
The software is based on python 3.8.20 and has been tested using the packages versions listed in requirements.txt but 
the version used are not a strict requirements. 
Pytorch being a bit capricious, it is recommended to use conda to install the packages in a virtual environment:
```bash
conda create -n pileup_analysis python=3.8
conda activate pileup_analysis
conda install jupyter matplotlib pandas scipy 
```
If you want to use the CPU only version of pytorch, you can install it with the following command:
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
```
But if you have a GPU and want to use it, you can install the GPU version of pytorch following the instructions on the 
official website: https://pytorch.org/get-started/locally/
An environment compatible with GPU computing are usually available on most computing clusters (on CEA computing cluster,
for example, you can use the "gpu" environment with the command 
```bash
module load /feynman/work/projets/cupid/scratch/mpageot/python_project/ML_denoising_env"
```
This operation might be tricky but increase a lot the speed of the computation as the optimization uses gradient 
descent and other GPU optimized operations. If you have a GPU, it is highly recommended to use it.

Then you create a `.env` file in the root directory of the software following the example provided in the `.env.example` 
file. The `.env` file is used to set the path to the data used for injection (ie raw binary file)

## Structure of the software
- `src`: 
    - `analysis.py`: contains the main code to perform the analysis and compute the Background Index.
    - `dataset.py`: contains the code to generate the dataset used for the analysis, reading binnary files and caching for
increased speed.
    - `simulation.py`: contains the code to simulate noise and pileup events to test the response of the filter. It also 
contains the code to generate the response of the detector to the bessel filter used during aquisition.
    - `plots.py`: contains the code to generate the figures of the paper.

- `utility`: 
    - `double_beta_spectrum.py`: contains the code to generate the double beta spectrum of 100Mo used to generate 
the montecarlo events as well as the associated spectrum of the r (relative amplitude) of pileup given that their 
energy is equal to the Q-value of the double beta decay.
    - `functions.py`: contains torch translation of certain numpy functions.

- `test`: contains `tutorial.ipynb`,a jupyter notebook tutorial to guide the user through the main steps of the analysis
and show how to use as well as all the scripts used to quantify the systematic uncertainties of the analysis.

