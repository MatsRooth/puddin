"""
Quick metatool script to get a portion of a larger pickled and gzipped dataframe as a csv file. 

    arguments: [path to pkl(.gz) file] <number of rows for output: default=10> <starting row index: default=0>
    
    example: 
        pile_tables/tmp/pile_00_Pile-CC_df.pkl.gz 50 20000
        --> will output csv of rows 20000-
"""
from pathlib import Path
from sys import argv

import pandas as pd


inpath = Path(argv[1]).resolve()
# inpath = Path('pile_tables/tmp/pile_pile-val_Pile-CC_df.pkl.gz').resolve()
print('source file:', inpath.relative_to(Path.cwd()))
fulldf = pd.read_pickle(inpath)
ix_col = 'index' if 'index' in fulldf.columns else 'text_id'
fulldf = fulldf.set_index(ix_col)

try:
    n = int(
        # 20
        argv[2]
    )
except IndexError:
    n = 10
print('number of rows:', n)
try:
    startix = int(
        # 1000
        argv[3]
    )
except IndexError:
    ix = 0
else:
    ix = min(len(fulldf) - n, startix)

print('starting at:', ix)

partial = fulldf.iloc[ix:ix+n, :]

outdir = inpath.parent.joinpath('partials').resolve()
if not outdir.is_dir():
    outdir.mkdir()
outpath = outdir.joinpath(f'{inpath.stem.split(".")[0]}{ix}plus{n}.psv')

partial.to_csv(outpath, sep='|')
print(f'Selected portion of {inpath.name} ({n} rows, {ix}-{ix+n}) '
      f'saved to {outpath.relative_to(Path.cwd())}')
