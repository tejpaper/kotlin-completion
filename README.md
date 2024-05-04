# Kotlin Code Completion with Phi-1 Transformer

## Description

This repository offers a solution for code completion in Kotlin projects using the Phi-1 Transformer model. It includes tools to extract Kotlin code, adapt the model, and fine-tune it on a custom dataset. Evaluate its performance on both Kotlin and Python code completion tasks.

The main approaches and results are described in the `analysis.ipynb` file. Additional README files are also present in the project. See `data` and `train` folders.

## Installation

```bash
git clone https://github.com/tejpaper/kotlin-completion.git
cd kotlin-completion
pip install -r requirements.txt
```

The data file for the Python code test is too big for GitHub. It must be downloaded separately by executing `data/python/download.py`.

## License

All Kotlin code is subject to the respective licenses according to [this](https://github.com/JetBrains/kotlin/blob/master/license/README.md) file.

Everything else is distributed under MIT.
