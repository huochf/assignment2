# Assignment #2 (CS 272 - Computer Vision Ⅱ, 2022-2023 Fall)

This code repository is for the course: CS 272 - Computer Vision Ⅱ, 2022-2023 Fall. Please follow instructions to setup environment and conduct your experiment.

## Requirements

* 4GB GPU memory for batch size128

## Setup Environment

```
conda create --name assignment2 python=3.9
conda activate assignment2
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
pip install nltk pycocotools opencv-python
```

## Download Dataset

run script to download.sh coco dataset

```
bash ./utils/download.sh
```

it will take a moment to download COCO 2014 (13G training set + 6G validation set) and its annotations (about 0.2G). After downloading, the dataset will show up directory ```./data/```.

## Preprocess the Dataset

We follow Karpathy Split to randomly select 5000 images for test, 5000 images for validation and the rest for training. Please run

```
python ./utils/KarpathySplit.py
```

The split file will show up in the folder ```data/annotations/```. After this, we need to build vocabulary. Please run

```
python ./utils/build_vocab.py
```

before you first run above command, you may need to download ```punkt```

```
python
>>> import nltk
>>> nltk.download('punkt')
```

then file ```data/vocab.pkl``` will be generated.

## Clone coco caption evaluation tools

```
cd ./utils
git clone https://github.com/sks3i/pycocoevalcap.git
```

## Task 1: Run through base model (30 pts)

just run

```
python ./train.py --config configs/config_base.ini
```

if you are in AI Cluster, just run

```
sbatch ./scripts/train_base.slurm
```

(you may need to redirect the path for logging and root directory)

One epoch can be trained within 30 minutes.  Model will converge to 0.66 (+-0.01) BLEU-1 after 5 epochs.

The logs and checkpoints are saved to directory ```outputs/base```.

After training, you can visualize results by running

```
python ./visualize.py --config configs/config_base.ini
```

results are saved to directory ```outputs/base/visualization/```.

Next, you need to try to improve the performance of base model by trying following techniques

* use more deeper and powerful backbone,
* adjust the hyper-parameter of the base model,
* use glove word vector to embed tokens,
* use beam search instead of greedy search,
* finetune backbone after training LSTM

## Task 2: Implement the attention model (30 pts)

You need to implement attention module described in figure 2(b) of [this paper](https://arxiv.org/abs/1612.01887), you can refer to existing codes in github, but you need to adapt them into this code framework cleverly. Here we given several steps you may need to do

1. build your encoder and decoder in folder ```./modeling/``` and their config file in ```./configs/```

2. import your model in ```train.py```, register their optimize and model (line73-90).

3. run

   ```
   python ./train.py --config configs/config_attention.ini
   ```

4. run

   ```
   python ./visualize.py --config <path-to-config>
   ```

   to show the attention maps.

attention model can achieve 0.64 BLEU-1 or higher after one epoch.

## Task 3: Try other attention models (40 pts)

Follow the suggestions in homework to adapt other attention mechanism.

## Notes on submission

you need to submit your codes except:

* all data in folder ```./data/```

* logs in folder ```./logs/```

* checkpoints in folder ```./outputs/```

  

what you need to submit should include:

* all files in folder ```./configs/```, ```./datasets/```, ```./modeling/```, ```./utils/```,  and files ```./train.py```, ```./eval.py```, ```./visualize.py``` ,
* all eval metrics files ```./outputs/<model_type>/<model_type>_eval_metric.json```,
* several images visualized using your model ```./outputs/<model_type>/visualization/*.jpg```.

## Acknowledgements

This code framework are based on this repository:

[Neural-Image-Captioning](https://github.com/SathwikTejaswi/Neural-Image-Captioning) 

# Question & Answering

1. version of gcc/g++ is too low, fail to install pycocotools.
First way, you can use gcc-10.2.0 which has been compiled in AI cluster.
Add two lines to the file ```/public/home/<your name>/.bashrc```
```
export PATH=/public/software/gcc/gcc-10.2.0/bin/:/public/software/gcc/gcc-10.2.0/lib64/:$PATH
export LD_LIBRARY_PATH=/public/software/gcc/gcc-10.2.0/lib/:$LD_LIBRARY_PATH
```
then run
```
source ./.bashrc
gcc -v
```

Or you can compile by yourself, follow [this instriction](https://blog.csdn.net/qq_36303832/article/details/119118519) to upgrade gcc in AI CLuster under user permissions.
First, you need to download gcc. If we want to use gcc-7.5.0 version, run
```
cd /public/home/<your name>/
mkdir gcc & cd gcc
wget http://ftp.gnu.org/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz
tar -zxvf ./gcc-7.5.0.tar.gz
cd ./gcc-7.5.0/
... ...
```
