# coding=utf-8

import argparse
import time
from pathlib import Path

import pandas as pd


def _main():
    args = _parse_args()
    dir_path = args.directory_path
    out_path = args.output_path
    print(f'Concatonating files in {dir_path}', time.ctime(
    ), sep='\n @ ', end='\n* * * * * * * * * * * * * *\n')

    for f in dir_path.glob('*.pkl*'):
        print(str(Path(*f.parts[-3:])))

    dfs_iter = (pd.read_pickle(f).assign(info_path=f)
                for f in dir_path.glob('*.pkl*'))

    combined = pd.concat(dfs_iter)
    combined = _make_categoricals(
        combined, ("code", "stem", "group", "type", "slice", "path")).sort_index()
    combined.to_pickle(out_path)

    print(f'Composite dataframe saved to {out_path}')


def _make_categoricals(df: pd.DataFrame,
                       cat_suff: tuple):

    cat_cols = df.columns.str.endswith(cat_suff)
    df.loc[:, cat_cols] = df.loc[:, cat_cols].astype(
        'string').astype("category")
    return df


def _parse_args():

    parser = argparse.ArgumentParser(
        description=(
            'Concatonate all pickled dataframes in a directory into a sinble composite file.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('directory_path',
                        type=Path,
                        help=(
                            'Path to directory of pickled dataframes to be concatonated.')
                        )

    parser.add_argument('output_path', type=Path,
                        help=('Path to save composite dataframe to. '
                              '*Outside* of the starting directory is recommended to avoid accidental recursion!. '
                              'Will be saved as pickle, so should have .pkl(.gz) extension.'))

    return parser.parse_args()


if __name__ == '__main__':
    _main()
