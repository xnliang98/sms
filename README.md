# SpanBERT + MSA
Code for NAACL 2022 paper "Modeling Multi-Granularity Hierarchical Features for Relation Extraction". 

## Requirements

transformers==3.0.2

python==3.7

pytorch==1.3.1

## Data and Pretrained model
TACRED dataset is not open access, so we provide the cached file for the training and testing of our model. (If needed, you can get the raw data from https://catalog.ldc.upenn.edu/LDC2018T24. The TACRED Revisted dataset can be obtained by https://github.com/DFKI-NLP/tacrev.)

We use SpanBert from https://huggingface.co/SpanBERT/spanbert-large-cased.

TODO: We will released our trained model soon.

## Evaluation
We provided log file of evaluation in `logs`.
You can evaluate our model by run:

```shell
chmod +x eval_tacred.sh; ./eval_tacred.sh
chmod +x eval_tacred_rev.sh; ./eval_tacred_rev.sh
```


## Training
We train our model on single NVIDIA V100 GPU about 4 hours.
You can train a new model by:

```shell
chmod +x run_spanbert.sh; ./run_spanbert.sh
```
