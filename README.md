# RAAN
Rare-Aware Attention Network for Image-Text Matching (Already submitted at Information Processing & Management 2022)
## Introduction
In this work, we propose a novel rare-aware attention network (RAAN), which explicitly explores and exploits the characteristics of rare content for tackling the long-tail effect.  
## Requirements and Installation
We recommended the following dependencies.      
* Python 3.8    
* [PyTorch 1.10.0](http://pytorch.org/)  
* [NumPy (>1.19.5)](http://www.numpy.org/)   
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)   
## Download data
The data files used are from [GSMN](https://github.com/CrossmodalGroup/GSMN).
## Training
    python train.py --data_name f30k_precomp --bi_gru --max_violation --lambda_softmax=20 --num_epochs=20 --lr_update=10 --learning_rate=.0002  --embed_size=1024 --batch_size=146
## Evaluation
    python test.py
