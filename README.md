# Deep Learning for small RNA mediated targeting

Micro RNAs (miRNAs) have a significant, regulatory role in key biological processes and diseases, 
by post-transcriptional gene expression modulation achieved by binding to target sites on messenger RNAs. 
Algorithmic prediction of the potential of a miRNA-target binding is hindered by the yet not properly addressed 
problem of class imbalance between the “few” actual binding sites (the positive class) and the overrepresented 
negative class (remaining regions). Here we provide an overview of general methods designed to deal with the 
class imbalance problem and evaluate their impact on this task. In addition, we propose a novel method called 
Instance hardness based label smoothing, which selects problematic regions for training and smoothens the labels 
based on the hardness of individual samples. Experimental results show that training convolutional neural networks 
with an arbitrary architecture using our approach significantly improves the area under the precision-recall curve 
(and other evaluation metrics), while notably reducing the required training time.

## Repository structure 

This repository contains the functions developed for the novel method "Instance hardness based label smoothing", with 
runnable examples in the `examples.ipynb` notebook. Apart from that, we provide utility functions building deep neural 
architectures also developed for the task of miRNA-target binding prediction.

```
dl_for_small_rna_mediated_targeting
│ 
│   examples.ipynb    
│
└───utils
│   │   data.py
│   │   label_smoothing.py
│   │   testing.py
│   │   visualization.py
│   │   
│   └───architectures
│       │   cnn.py
│       │   inception.py
│       │   resnet.py
│
└───models
│   │   resnet_small_committee.h5
│   │   resnet_small_scout.h5
│   │   scout_resnet.h5
│
└───datasets
│   │   train_set_1_100_CLASH2013_paper.tsv
│   │   ...
│   │   train_pool_IH_committee.csv
│
└───archive
    └───experiments
        │   ...
```

`examples.ipynb`

notebook containing runnable examples of our method - the instance hardness based label smoothing in 2 versions.

`utils`

utils is a python module which contains all functions necessary to use the methods we developed

`utils.architectures`

this submodule of utils contains functions which create the 3 architectures we worked with - CNN "optimized", ResNet and Inception.

`models`

directory with pretrained models from the examples notebook

`datasets`

contains datasets necessary to run the "examples" notebook, with the rest of the datasets (used in this work) available 
in the [ML-Bioinfo-CEITEC/miRBind repo](https://github.com/ML-Bioinfo-CEITEC/miRBind/tree/main/Datasets). File 
*train_pool_ih_committee.csv* contains IH scores produced by a committee and is included as a reference and for 
one of the examples.

`archive`

Most of the notebooks in archive are not runnable. Their sole purpose is to present the experiments that shaped 
our final solution, with results presented in the text of the master thesis. During experimentation, we ran all 
experiments on multiple instances of Google Colab and on our own computers, aggregating the results into the thesis. 
In the "archived" notebooks we omit preprocessing, TPU / GPU initialization and other repeating necessities, which can be 
seen in the examples.ipynb notebook.
