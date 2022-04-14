[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

# This repository is deprecated

Please take a look at our latest BirdNET repository with updated models, tools and interfaces: [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) 

# BirdNET Soundscape Analysis
By [Stefan Kahl](https://github.com/kahst), [Shyam Madhusudhana](https://www.birds.cornell.edu/brp/shyam-madhusudhana/), and [Holger Klinck](https://www.birds.cornell.edu/brp/holger-klinck/)

For more information regarding the project visit: https://birdnet.cornell.edu/

Please cite as:

```
@phdthesis{kahl2019identifying,
  title={{Identifying Birds by Sound: Large-scale Acoustic Event Recognition for Avian Activity Monitoring}},
  author={Kahl, Stefan},
  year={2019},
  school={Chemnitz University of Technology}
}
```

You can download the PDF here: [https://monarch.qucosa.de](https://nbn-resolving.org/urn:nbn:de:bsz:ch1-qucosa2-369869)

For a list of supported species visit: [https://birdnet.cornell.edu/species-list/](https://birdnet.cornell.edu/species-list/)

<b>Please also take a look at our newest repository:</b> [BirdNET for Tensorflow Lite](https://github.com/kahst/BirdNET-Lite)

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Introduction
How can computers learn to recognize birds from sounds? The Cornell Lab of Ornithology and the Chemnitz University of Technology are trying to find an answer to this question. Our research is mainly focused on the detection and classification of avian sounds using machine learning – we want to assist experts and citizen scientist in their work of monitoring and protecting our birds.

This repository provides the basic BirdNET recognition system to detect avian vocalizations in soundscape recordings. We also provide a list of files that were used for training and testing (filenames are indicative of the individual file IDs). See ```BirdNET_1000_Dataset.csv```for more details.

<b>If you have any questions or problems running the scripts, don't hesitate to contact us.</b>

Contact:  [Stefan Kahl](https://github.com/kahst), [Chemnitz University of Technology](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.kahl@informatik.tu-chemnitz.de

## Installation
This is a Theano/Lasagne implementation in Python for the identification of hundreds of bird species based on their vocalizations. This code is tested using Ubuntu 18.04 LTS but should work with other distributions as well. Python 2 and 3 are supported. See <i>Installation (Docker)</i> to install BirdNET inside a docker container.

1. Clone the repository:

```
git clone https://github.com/kahst/BirdNET.git
```

2. Install requirements:

```
cd BirdNET
pip install –r requirements.txt
```

You might need to add the full path to the requirements.txt in case pip throws an error:

```
pip install -r /path/to/requirements.txt
```

<i>These versions of required packages are known to work: NumPy 1.14.5, SciPy 1.0.0, Librosa 0.7.0, Future 0.17.1</i>

3. Librosa required audio backend (FFMPEG):

```
sudo apt-get install ffmpeg
```

4. Download model snapshot (300 MB):

```
sh model/fetch_model.sh
```

5. Install Lasagne/Theano:

```
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

6. Install BLAS:

```
sudo apt-get install libblas-dev liblapack-dev
```

<i>You might have to add 'sudo' in front of the 'pip' command when admin privileges are required.</i>

7. Install GPU support (optional, but significantly faster):

Before installing <i>libgpuarray</i>, you need to install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

You also might need to install Cython and cmake:

```
sudo apt-get install cmake
pip install Cython 
```

After that, you can install libgpuarray with:

```
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray

cd <dir>
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
make install
cd ..

python setup.py build
python setup.py install

sudo ldconfig
```

<i>Again, you might need to add 'sudo' before install commands if admin privileges are required.

Please refer to the [Theano](http://deeplearning.net/software/theano/install_ubuntu.html) and [libgpuarray](http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install) install instructions if you encounter errors during install or execution.</i>

## Installation (Docker)

First, you need to install docker. You can follow the official [install guide](https://docs.docker.com/v17.09/engine/installation/) or run:

```
sudo apt-get update

sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get update
sudo apt-get install docker-ce
```

After that, clone the repository and build the BirdNET docker container with:

```
git clone https://github.com/kahst/BirdNET.git
cd BirdNET
sudo docker build -t birdnet .
```

When finished, you can run the container and start the analysis in CPU mode (see Usage (Docker)).

If you want to use our GPU docker image to run BirdNET in GPU mode, you need to install <i>nvidia-docker</i> before building the <i>Dockerfile-GPU</i>. Follow the official [install guide](https://github.com/NVIDIA/nvidia-docker) or run:

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

After that, build the GPU image with:

```
sudo docker build -f Dockerfile-GPU -t birdnet-gpu .
```

All you need are GPU drivers that support CUDA 9.2 or higher. You do not need to install CUDA or cuDNN on the host system.

## Usage
BirdNET is an artificial neural network that can detect bird vocalizations in field recordings. This implementation runs in CPU mode and does not require specialized hardware (you can run the script in GPU mode if you have a CUDA-enabled GPU available). A number of optional settings can be provided when executing the analysis script. Here are some examples for basic usage:

Analyze all '.wav'-files in a directory:

```
python analyze.py --i example/
```

Analyze a single file with known recording location:

```
python analyze.py --i example/Soundscape_1.wav --lat 42.479 --lon -76.451
```

The input arguments include:

```
--i: Path to input file or directory.
--o: Path to output directory. If not specified, the input directory will be used.
--filetype: Filetype of soundscape recordings. Defaults to 'wav'.
--results: Output format of analysis results. Values in ['audacity', 'raven']. Defaults to 'raven'.
--lat: Recording location latitude. Set -1 to ignore.
--lon: Recording location longitude. Set -1 to ignore.
--week: Week of the year when the recordings were made. Values in [1, 48]. Set -1 to ignore.
--overlap: Overlap in seconds between extracted spectrograms. Values in [0.0, 2.9].
--spp: Combines probabilities of multiple spectrograms to one prediction. Defaults to 1.
--sensitivity: Sigmoid sensitivity; Higher values result in lower sensitivity. Values in [0.25, 2.0]. Defaults to 1.0.
--min_conf: Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.'
```

Output formats support Raven and Audacity, but both formats are text-based and machine-readable.

## Usage (Docker)

In order to pass a directory that contains your audio files to the docker file, you need to mount it inside the docker container with <i>-v /my/path:/mount/path</i> before you can run the container. 

You can run the container for the provided example soundscapes with:

```
sudo docker run -v $PWD/example:/audio birdnet --i audio
```

You can adjust the directory that contains your recordings by providing an absolute path:

```
sudo docker run -v /path/to/your/audio/files:/audio birdnet --i audio
```

You can pass all aforementioned command line arguments (e.g. lat, lon, week) to the analysis script when starting the docker container:

```
sudo docker run -v /path/to/your/audio/files:/audio birdnet --i audio --lat 42.479 --lon -76.451 --week 12
```

If you built the GPU docker image, you can run the analysis in GPU mode by using <i>docker run --gpus all</i> instead:

```
sudo docker run --gpus all -v /path/to/your/audio/files:/audio birdnet-gpu --i audio --lat 42.479 --lon -76.451 --week 12
```

<i>You might not need 'sudo' before 'docker run' if your user is member of the docker group</i>

## Sponsors

This project is supported by Jake Holshuh (Cornell class of ’69). The Arthur Vining Davis Foundations also kindly support our efforts.

The European Union and the European Social Fund for Germany partially funded this research. This work was also partially funded by the German Federal Ministry of Education and Research in the program of Entrepreneurial Regions InnoProfileTransfer in the project group localizeIT (funding code 03IPT608X)
