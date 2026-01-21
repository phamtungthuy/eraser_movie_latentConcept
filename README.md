# Latent Concept Explanation for Sentiment Classification Task

This is the codebase for our EMNLP 2024 paper: [Latent Concept-based Explanation of NLP Models](https://aclanthology.org/2024.emnlp-main.692/).

## Table of Contents
* [Usage](#Usage)
* [Project Structure](#Project-structure)
* [Run Code](#Run-code)
* [Citation](#Citation)

## Usage
Generating explanations for classification predictions based on latent concepts.

## Project Structure
- `data/`: Contains the datasets used for analysis.
- `src/`: Source code for the analysis.
  * `src/clustering`: Extract the latent concepts from training dataset
  * `src/classifier_mapping`: Train a conceptMapper
  * `src/IG_backpropagation`: Discover the salient tokens for input instances
  * `src/generate_explanation_files`: Get the predicted label and generate the file with the 'predicted label',
'sentence_idx' and 'word_idx'
  * `src/concept_mapper`: Map tokens to the latent concepts
- `scripts/`: Includes scripts for running the source code.
  * `src/train_classifier`: Fine-tune BERT for sentiment classification

## Prerequisites: Fine-tune BERT Model

**Important:** Before running the LACOAT pipeline, you need a fine-tuned BERT model for your classification task. The base BERT model (`google-bert/bert-base-cased`) does NOT have a trained classification head, so using it directly will produce random predictions.

### Step 1: Train the Sentiment Classifier

Run the training script to fine-tune BERT on the movie reviews dataset:

```bash
./scripts/train_classifier/train_bert_sentiment.sh
```

Or run directly with custom parameters:

```bash
python src/train_classifier/train_bert_sentiment.py \
    --train-file data/movie_train.json \
    --model-name google-bert/bert-base-cased \
    --output-dir trained_models/bert_sentiment \
    --epochs 3 \
    --batch-size 16
```

### Step 2: Update Configuration

After training, update `config.env` to use your fine-tuned model:

```bash
# config.env
MODEL=trained_models/bert_sentiment   # or absolute path to your trained model
```

Now you can proceed with the LACOAT pipeline below.

## Run Code
### Training Phase
1. Extract Latent Concepts:
   * Run `scripts/train_set/clustering/clustering_base_work.sh`
   * Run `scripts/train_set/clustering/clustering_conceptX.sh`

2. Train a ConceptMapper:
   * Run `scripts/train_set/classifier_mapping/generate_csv_file.sh`
   * Run `scripts/train_set/classifier_mapping/split_dataset.sh`
   * Run `scripts/train_set/classifier_mapping/logistic_regression.sh`
   * Run `scripts/train_set/classifier_mapping/get_prediction_stat.sh`

3. Discover the Salient Tokens:
   * Run `scripts/train_set/IG_backpropagation/ig.sh`

4. Get the Prediction:
   * Run `scripts/train_set/generate_explanation_files/generate_CLS_explanation.sh`
   * Run `scripts/train_set/generate_explanation_files/generate_all_tokens_explanation.sh`
   * Run `scripts/train_set/generate_explanation_files/generate_IG_explanation.sh`

### Inference Phase
1. Extract Latent Concepts:
   * Run `scripts/dev_set/clustering/clustering_base_work.sh`
   * Run `scripts/dev_set/clustering/clustering_conceptX.sh`

2. Discover the Salient Tokens:
   * Run `scripts/dev_set/IG_backpropagation/ig.sh`

3. Get the Prediction:
   * Run `scripts/dev_set/generate_explanation_files/generate_CLS_explanation.sh`
   * Run `scripts/dev_set/generate_explanation_files/generate_all_tokens_explanation.sh`
   * Run `scripts/dev_set/generate_explanation_files/generate_IG_explanation.sh`

4. Map Tokens to the Latent Concepts:
   * Run `scripts/dev_set/concept_mapper/match_representation.sh`
   * Run `scripts/dev_set/concept_mapper/logistic_regression.sh`


## Citation
```
@inproceedings{yu-etal-2024-latent,
    title = "Latent Concept-based Explanation of {NLP} Models",
    author = "Yu, Xuemin  and
      Dalvi, Fahim  and
      Durrani, Nadir  and
      Nouri, Marzia  and
      Sajjad, Hassan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.692",
    pages = "12435--12459",
 }
```