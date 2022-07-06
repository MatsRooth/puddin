# %% [markdown]
# # Confirm Text/Document ID Coverage
# All (Pcc) texts in the original pile files as saved in `pile_tables/raw/` should show up in either the exclusions file for that data group (i.e. as row in the exclusions dataframe) or the final conllu directory (i.e. as a document ~ `new_doc` comment in one of the conllu files there).
#
# This notebook works through collecting the IDs found in each of these 3 places and comparing the resulting objects to confirm nothing has been lost.
#
# _One possible concern however, is that the exclusions files may be inaccurate. That is, there are documents that were erroneously marked as `fail`s and skipped during parsing, but then successfully completed in a following reparse. So really, it's just the comparison of the raw dataframes and the conllu files that matter, and any that are missing should be in the exclusions dataframe. After this comparison is drawn, documents/texts added to the exclusions in error (i.e. IDs that actually do have parsed sentences in a conllu file) should be removed from the exclusions dataframe_
#
# If there are raw text IDs not found in the conllu files or the exclusions, the final (top level `pile_tables/`) and temporary (`pile_tables/tmp/`) should be searched and/or the dataframes in `slices/[data group]/tmp/` should be compared with "final" slices in `slices/[data group]/`.

# %%
# coding=utf-8
#!/home/arh234/.conda/envs/dev-sanpi/bin/python

import pandas as pd
import pyconll

from pathlib import Path
# from datetime import datetime
# tstamp = datetime.fromtimestamp
DATA_DIR = Path('/share/compling/data/puddin')
DATA_GRP = 'val'
DF_NAME = f'pile_{DATA_GRP}_Pile-CC_df.pkl.gz'


# %% [markdown]
# ## Load meta info dataframe

# %%
meta = pd.read_csv(DATA_DIR.joinpath('completed-puddin_meta-index.csv'))
meta.head()

# %% [markdown]
# For each row (i.e. slice) compare the text ids found in the files at the following paths: raw, final, conllu. Make sure any missing from conllu are in exclusions dataframe.
#
# _Should probably group by exclusions path/data group/original data source column_

# %%


def newdocs_iter(conll_dir):
    for f in conll_dir.glob('*.conllu'):
        for s in pyconll.iter_from_file(str(f)):
            try:
                doc_id = s.meta_value('newdoc id')
            except KeyError:
                continue
            recons_raw_id = '_'.join([x.replace('x', '')
                                     for x in doc_id.split('_')[0:5:2]])
            yield recons_raw_id


# %%
findfs = []
rawdfs = []
problems = []
for grp, mdf in meta.groupby('exclusions_path'):
    excl_path = DATA_DIR.joinpath(grp)
    # print(excl_path)
    # sanity checks
    if len(mdf.final_df_path.unique()) != 1:
        print('WARNING! different paths showing for final df')
    if len(mdf.origin_filepath.unique()) != 1:
        print('WARNING! different paths showing for source file')
    findf_path = DATA_DIR.joinpath(mdf.final_df_path.iloc[0])
    # print(findf_path)
    rawdf_path = DATA_DIR.joinpath(f'pile_tables/raw/{findf_path.name}')
    # print(rawdf_path)
    conll_dir = DATA_DIR.joinpath(Path(mdf.conllu_path.iloc[0]).parent)
    # print(conllu_dir)
    # print('...........\n')

    # * now check text ids for each
    rawdf = pd.read_pickle(rawdf_path)
    findf = pd.read_pickle(findf_path)
    excldf = pd.read_pickle(excl_path)
    conllu_ids = set(newdocs_iter(conll_dir))

    findf = findf.assign(late_excl=~findf.text_id.isin(conllu_ids))
    rawdf = rawdf.assign(init_excl=~rawdf.text_id.isin(findf.text_id))
    all_excl_ids = findf.text_id.loc[findf.late_excl] + \
        rawdf.text_id.loc[rawdf.init_excl]
    unaccounted = [i for i in all_excl_ids not in excldf.text_id]
    print(unaccounted)
    problems.append(unaccounted)

# ? turn this into a dataframe with raw text id as index and columns as booleans for e.g. 'in final df', 'in exclusions', 'in conllu parse', etc?


# %% [markdown]
# ## 1. Get the original text IDs from the raw dataframes

# %%
raw_ex = pd.read_pickle(DATA_DIR.joinpath(
    f'pile_tables/raw/{DF_NAME}'))
# raw_ex


# %% [markdown]
# ## 2. Get the text IDs from the final dataframe

# %%
fin_ex = pd.read_pickle(DATA_DIR.joinpath(f'pile_tables/{DF_NAME}'))
fin_ex


# %%
orig_excl = raw_ex.loc[~ raw_ex.text_id.isin(fin_ex.text_id), 'text_id']

# %% [markdown]
# ## 3. Get the document (text) IDs from the finalized conllu files

# %%
not_in_conllu = [i for i in fin_ex.text_id if i not in doc_ids]
not_in_conllu

# %%
fin_ex.text_id
