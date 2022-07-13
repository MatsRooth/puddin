# coding=utf-8
#!/home/arh234/.conda/envs/dev-sanpi/bin/python
'''
Confirm Text/Document ID Coverage

All (Pcc) texts in the original pile files as saved in `pile_tables/raw/`
should show up in either the exclusions file for that data group
(i.e. as row in the exclusions dataframe) or the final conllu directory
(i.e. as a document ~ `new_doc` comment in one of the conllu files there).

This notebook works through collecting the IDs found in each of these 3 places
and comparing the resulting objects to confirm nothing has been lost.

_One possible concern however, is that the exclusions files may be inaccurate.
That is, there are documents that were erroneously marked as `fail`s
and skipped during parsing, but then successfully completed in a following reparse.
So really, it's just the comparison of the raw dataframes and the conllu files
that matter, and any that are missing should be in the exclusions dataframe.
After this comparison is drawn, documents/texts added to the exclusions in error
(i.e. IDs that actually do have parsed sentences in a conllu file) should be
removed from the exclusions dataframe_

If there are raw text IDs not found in the conllu files or the exclusions,
the final (top level `pile_tables/`) and temporary (`pile_tables/tmp/`)
should be searched and/or the dataframes in `slices/[data group]/tmp/`
should be compared with "final" slices in `slices/[data group]/`.
'''
# %%

# import re
# import subprocess as sp
from pathlib import Path

import pandas as pd

from pull_ids_from_conll import conllu_id_iter, reconstruct_raw_iter
# import pyconll

# %%
# from datetime import datetime
# tstamp = datetime.fromtimestamp
DATA_DIR = Path('/share/compling/data/puddin')
DATA_GRP = 'val'
DF_NAME = f'pile_{DATA_GRP}_Pile-CC_df.pkl.gz'

# ## Load meta info dataframe

meta = pd.read_csv(DATA_DIR.joinpath('completed-puddin_meta-index.csv'))
# meta.head()


# For each row (i.e. slice) compare the text ids found in the files at
# the following paths: raw, final, conllu. Make sure any missing
# from conllu are in exclusions dataframe.
#
# _Should probably group by exclusions path/data group/original data source column_

findfs = []
rawdfs = []
problems = []
# %%
for grp, df in meta.groupby('exclusions_path'):
    # TODO: temporary, for debugging interactively w/o reloading
    if grp.find('test') < 0:
        continue
    else:
        excl_path = DATA_DIR.joinpath(grp)
        mdf = df
        break

# %%
# sanity checks
if len(mdf.final_df_path.unique()) != 1:
    print('WARNING! different paths showing for final df')
if len(mdf.origin_filepath.unique()) != 1:
    print('WARNING! different paths showing for source file')
findf_path = DATA_DIR.joinpath(mdf.final_df_path.iloc[0])
# print(findf_path)
rawdf_path = DATA_DIR.joinpath(f'pile_tables/raw/{findf_path.name}')
print(f'Loading initial/raw dataframe:\n    {rawdf_path}\n...')
rawdf = pd.read_pickle(rawdf_path)
raw_ids = rawdf.text_id
conll_dir = DATA_DIR.joinpath(Path(mdf.conllu_path.iloc[0]).parent)
print('directory of conllu files to be searched:\n   ', conll_dir)

# TODO: reindent when done debugging
# %%
parsed_doc_ids = conllu_id_iter(conll_dir, 'doc')
raw_texts = rawdf.rename(columns={'raw': 'raw_text', 'text_id': 'raw'})

# TODO : finish this code; update def reconstruct_raw_iter() too
# %%
parsed_texts = pd.DataFrame(reconstruct_raw_iter(parsed_doc_ids))
texts_df = raw_texts.reset_index().set_index(
    'raw').join(parsed_texts.set_index('raw'))
# for raw_match in reconstruct_raw_iter(parsed_doc_ids):
#     raw_id = '_'.join(raw_match.groups())
#     # status_df.loc[raw_id, 'parsed_id'] = raw_match.string
# %%
findf = pd.read_pickle(findf_path)
excldf = pd.read_pickle(excl_path)

# %%
texts_df = texts_df.assign(success=~texts_df.parsed.isna(),
                           init_excl=~texts_df.index.isin(findf.text_id))
texts_df = texts_df.assign(late_excl=~(texts_df.success | texts_df.init_excl))

texts_df = texts_df.assign(problem = ~ (texts_df.index.isin(excldf.text_id) | texts_df.parsed.isin(excldf.text_id) | texts_df.success))
problem_texts = texts_df.loc[texts_df.problem, :]
# %%
problem_texts.to_csv(DATA_DIR.joinpath('problem_texts.psv'), sep='|')
problems_json = problem_texts.to_json(DATA_DIR.joinpath('problem_texts.json'), orient='index', indent=4)

# %%
