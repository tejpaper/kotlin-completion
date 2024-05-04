There are two ways to start fine-tuning a model from the beginning. It is not necessary to do so, as the `model` folder already contains the final model weights.

### Google Colab

To run the code on colab it is expected to have access to the Google Drive on which the following will be located:

1. `data/kotlin/train.csv`
2. `data/kotlin/dev.csv`
3. `tunekit` package
4. A `cache` folder to store checkpoints

### Kaggle

To run training on kaggle, the notebook must have access to two datasets:

1. **kotlin-completion** with the contents of the `data` folder
2. **tunekit** with the contents of `tunekit` package and `cache` folder
