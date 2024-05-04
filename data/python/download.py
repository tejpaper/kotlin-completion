import os
import requests
import tarfile

import pandas as pd
from tqdm import tqdm


def py2string(path: str) -> pd.Series:
    for encoding in ('utf-8', 'iso-8859-1'):
        try:
            with open(path, 'rb') as file:
                return pd.Series(dict(
                    path=path,
                    code=file.read().decode(encoding),
                    encoding=encoding))
        except UnicodeDecodeError:
            continue  # next iteration

    raise RuntimeError(f'Could not decode file:\n{path}.')


def main() -> None:
    url = 'https://files.srl.inf.ethz.ch/data/py150_files.tar.gz'
    tar_name = 'py150_files.tar.gz'

    if not os.path.exists(tar_name):
        response = requests.get(url, stream=True)

        with open(tar_name, 'wb') as tar:
            for data in tqdm(response.iter_content(chunk_size=20), desc='Downloading py150'):
                tar.write(data)

    python50k = 'python50k_eval.txt'
    data_tar = 'data.tar.gz'

    if not os.path.exists(python50k) or not os.path.exists(data_tar):
        with tarfile.open(tar_name) as tar:
            tar.extract(python50k)
            tar.extract(data_tar)

    files_dir = 'data'

    if not os.path.exists(files_dir):
        with tarfile.open(data_tar) as tar:
            tar.extractall(members=tqdm(tar, desc='Extracting code'))

    with open(python50k) as file:
        df50k = pd.DataFrame(dict(path=file.read().splitlines()))

    tqdm.pandas(desc='Compiling csv')
    df50k = df50k.path.progress_apply(py2string)
    df50k.to_csv('test.csv', index=False)


if __name__ == '__main__':
    main()
