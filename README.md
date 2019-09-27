# BirdNET
Soundscape analysis with BirdNET. For more information regarding the project visit: https://birdnet.cornell.edu/

By [Stefan Kahl](https://github.com/kahst), [Shyam Madhusudhana](https://www.birds.cornell.edu/brp/shyam-madhusudhana/), and [Holger Klinck](https://www.birds.cornell.edu/brp/holger-klinck/)

## Introduction
How can computers learn to recognize birds from sounds? The Cornell Lab of Ornithology and the Chemnitz University of Technology are trying to find an answer to this question. Our research is mainly focused on the detection and classification of avian sounds using machine learning – we want to assist experts and citizen scientist in their work of monitoring and protecting our birds.

This repository provides the basic BirdNET recognition system to detect avain vocalizations in soundscape recordings.

<b>If you have any questions or problems running the scripts, don't hesitate to contact us.</b>

Contact:  [Stefan Kahl](https://github.com/kahst), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.kahl@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

## Installation
This is a Thenao/Lasagne implementation in Python for the identification of hundreds of bird species based on their vocalizations. This code is tested using Ubuntu 16.04 LTS but should work with other distributions as well.

1. Clone the repository:

```
git clone https://github.com/kahst/BirdNET.git
```

2. Install requirements:

```
cd BirdNET
pip install –r requirements.txt
```

3. Install Lasagne/Theano:

```
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Usage

Analyze all '.wav'-files in a directory:

```
python analyze.py --i example/
```

Analyze a single file with known recording location:

```
python analyze.py --i example/Soundscape_1.wav --lat 42.479 --lon -76.451
```
