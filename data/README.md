### Kotlin

The data for training the model can be obtained from any sufficiently large and open repository written in Kotlin. To do this, this repository should be cloned into the `kotlin/kotlin-master` folder, after which `kotlin/compile.py` can generate three CSV files.

### Python

To evaluate the model on Python data, the py150 dataset (which is also used in [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main)) was used as a basis. The original dataset was prioritized because it does not limit the perception of the problem and allows to be reimagined for use with encoder architecture for the mask filling task.

`python/download.py` can be used to download and compile a test data file.