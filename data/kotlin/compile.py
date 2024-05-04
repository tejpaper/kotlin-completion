import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def kt2string(path: str) -> pd.Series:
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
    target_dir = 'kotlin-master'

    kt_files = [
        os.path.join(folder, file)
        for folder, _, files in os.walk(target_dir)
        for file in files if file.endswith('.kt')
    ]

    tqdm.pandas(desc='Compiling csv')
    df = pd.DataFrame(dict(path=kt_files))
    df = df.path.progress_apply(kt2string)
    df = df[df.code != '']

    random_seed = 1337
    df, test_df = train_test_split(df, test_size=20_000, random_state=random_seed, shuffle=True)
    train_df, dev_df = train_test_split(df, test_size=5000, random_state=random_seed, shuffle=True)

    train_df.to_csv('train.csv', index=False)
    dev_df.to_csv('dev.csv', index=False)
    test_df.to_csv('test.csv', index=False)


if __name__ == '__main__':
    main()
