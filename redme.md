# MPCD: A multi-task graph Transformer for molecule property prediction by integrating common and domain knowledge 



![The framework of MPCD](https://github.com/XIXIYOUNG2018/MPCD/blob/main/framework.pdf)

MPCD is a newly developed method for evaluating the predictive performance on ADMET, physicochemical, and activity cliff compounds of machine learning models.







# Environment
## Install via conda yaml file 
```
conda env create -f environment.yml
conda activate mpcd 
```
## Install manually

```
conda create -n resgen python=3.8
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pyg -c pyg
conda install -c conda-forge rdkit
conda install biopython -c conda-forge
conda install pyyaml easydict python-lmdb -c conda-forge
```

# Use existing model and data!
Here is a quick use of validation. Using the following command:


```
sh examples/val/val.sh
```

Then the output will be:






# Data

## Download data
The pretraining data for MPCD is ChEMBL: https://www.ebi.ac.uk/chembl/.

The fine-tuning datasets include two datasets:

1. The ADMET and physicochemical datasets from ADMETlab2.0: https://admetmesh.scbdd.com/
2. The activity cliffs datasets from MoleculeACE: https://github.com/molML/MoleculeACE

Download and put them under file: dataset. You can name them by customer name.


## Data pre-process

After download data with csv format, you should pre-process them into pickle file and split them into train, test, and validation set.

```
python data_process/smiles_to_graph.py
```

# Train

MPCD is first pre-trained with ChEMBL dataset in dataset/ChEMBL.

## Pre-training with ChEMBL dataset
```
sh examples/pre_train.sh
```
## Fine-tuning

The training process and configs are released as train.sh, the following command is an example of how to train a model.
The command in train.sh includes the training process of ADMET, physicochemical, and activity cliffs datasets.

```
sh examples/train.sh
```
Also, the validation command includes the Python file and configs of  ADMET, physicochemical datasets and activity cliffs datasets
# Validataion

Evaluate the performance of MPCD.
```
sh examples/val.sh
```

Also, you can skip the pre-training， and directly use the pre-trained .cpk file to load the parameter.


Note that MPCD can be ready for any customer data set, as long as you use the propoess_data.py to transform your data into pickle format and load the
pretraining model. Then follow the same fine-tuning process.

# License
MPCD is under MIT license. For use of specific models, please refer to the model licenses found in the original packages.

