# BERTCRF

## Environment Setup
* Install python 3.9
* Cuda 11.6

## transform_data.py
First step is to run transform_data.py with a parameter from the following list [train, dev, test]. 
This file transforms data from given dataset to a format that will be readable from our model.
In order to run this file you have to run the following commands:

  * pip install mendelai-brat-parser
  * pip install smart_open
  * pip install nltk

## BERT_CRF_model.py
This is the main code that train of model happens.
In order to run this file you have to run the following commands:

  * pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
  * pip install transformers
  * pip install absl-py
  * pip install six
  * pip install protobuf==3.19.0
  * pip install wrapt
  * pip install opt_einsum
  * pip install gast
  * pip install astunparse
  * pip install termcolor
  * pip install flatbuffers
  * pip install scikit-learn
  * pip install sentence-splitter
  
  ## evaluation.py
  This file may run standalone if a model is already saved (system_best_epoch.pth.tar file exists).
  In order to achieve this you have to comment line 10: 
  ```
  def evaluate(model, class_dict, inv_class_dict, test_batches, use_cuda, gpu_device):
  ```
  and uncomment lines 13-15:
  ```
  # if __name__ == '__main__':
  #     from initialize import *
  #     model, class_dict, inv_class_dict, test_batches, use_cuda, gpu_device = my_initialize()
  ```
