# coding=utf-8

# # %% [markdown]
# # Notebook to investigate getting stats for puddin files

# %%

# > imports
import json
import pandas as pd
import pyconll
import sys
import time

from collections import namedtuple
from itertools import islice
from pathlib import Path
from pprint import pprint

# > constants
# local machine uses softlinks to mimic cluster architecture
DATA_DIR = Path('/share/compling/data/puddin')
STATS_DIR = DATA_DIR.joinpath('info', 'stats')
# TODO: remove sentence cap when finished developing!
CAP = 1000

# %% [markdown]
# ### Select dataset and load info dataframe

# %%


def describe_puddin(data_target: str = ''):
    start = time.perf_counter()
    # > read in parsing info collected at parsing completion and cleaned up afterwards
    # >   this dataframe contains relevant file size/timestamps and processing time info
    # >   indexed flatly by slice: i.e. single dataframe for all slices, regardless of data group
    # ^ can either add newly gathered conllu/slice descriptive stats to this flat structure
    # ^ ~OR~ take the relevant info for each conllu file/slice from here and add it to each group's individual df output below
    # ^ 🤔 👉suspect former will be required
    # TODO: see how large the original parsing info df is and see if it can handle adding the number of columns in the grp_by_conllu dfs or if it will make it too large to be handled easily
    parsing_slice_df_path = DATA_DIR.joinpath('info',
                                              'completed-puddin_meta-index.pkl')
    if parsing_slice_df_path.is_file():
        parsing_by_slice_df = pd.read_pickle(parsing_slice_df_path)
    else:
        sys.exit('Parsing info cannot be found. '
                 f'Invalid path: {parsing_slice_df_path}. Terminating.')

    valid_texts_info_dir = DATA_DIR.joinpath('info', 'validation_by_group',
                                             'status-overview')
    if not valid_texts_info_dir.is_dir():
        sys.exit('Validation outcomes could not be found. '
                 f'Invalid path:{valid_texts_info_dir}. Terminating.')
    grp_dicts = []
    # > loop through original parsing info by slice
    for data_grp, slice_info in parsing_by_slice_df.groupby('data_origin_group'):
        #! this data_grp value is just the group as a category, not a string
        # TODO: See if updated parsing info on cluster has more complete info, such as conll dir (consider adding them?)
        # ? why exactly am I pulling from the validation status info,
        # ?     which is indexed by *all* (even excluded) texts?
        # ?  As opposed to the slice-indexed dataframe output from the original parsing?
        #   ^ this could work as a way of making sure stats are gathered for validated data only?
        print(data_grp)
        # print(slice_info.sample(min(len(slice_info), 6)))
        # TODO: Finish this? What was this intended to do?

    for group_info_file in valid_texts_info_dir.glob(f'{data_target}*status-info*pkl*'):

        #! for testing --- or maybe not? use for slurm array job?
        if data_target and group_info_file.stem.startswith(data_target):
            continue
        data_grp = group_info_file.stem.split('_', 1)[0]
        group_data_dir = DATA_DIR.joinpath(f'{data_grp}.conll')
        if not group_data_dir.is_dir():
            # > do not expect to have all data locally,
            # > so for local devel just continue when dir does not exist
            # // sys.exit(
            # //     f'ERROR: cannot find conllu dir for {data_code}\n>> Program Terminated.')
            print(f'Cannot find {group_data_dir}. Skipping.')
            continue

        group_stats_dir = STATS_DIR.joinpath(data_grp)
        conllu_dicts = []

        valid_by_text_df = pd.read_pickle(group_info_file)
        # > `group_info` is indexed by raw text id, thus conllu_stems repeat and are sometimes null, for exception cases
        # > group_info.conll_id will match `D_id` later
        # > group_info.slice & group_info.conll_id are formatted slightly differntly than group_info.conllu_stem:
        #                                    conll_id          slice     conllu_stem
        #    raw_id
        #    pcc_val_00001  pcc_eng_val_1.0001_x00001  pcc_eng_val_1  pcc_eng_val-01
        #    pcc_val_00002  pcc_eng_val_1.0002_x00002  pcc_eng_val_1  pcc_eng_val-01
        #    pcc_val_00003                       <NA>           <NA>            <NA>
        #    pcc_val_00004                       <NA>           <NA>            <NA>
        #    pcc_val_00005  pcc_eng_val_1.0003_x00005  pcc_eng_val_1  pcc_eng_val-01
        # > group_info.text_altered indicates whether the parsed text differs at all from the text round in the original Pile source jsonlines files
        # TODO: index group_info by conll_id to pull info for altered text by doc

        # * Get conllu path from info dataframe
        # > filter for only texts that *have* a conllu file (i.e. ~conllu_stem.isna())
        parsed_info = valid_by_text_df.loc[valid_by_text_df.success, :]
        for stem in parsed_info.conllu_stem.unique():
            conllu_path = group_data_dir.joinpath(f'{stem}.conllu')
            if not conllu_path.is_file():
                # and conllu_path.stat().st_size > 0:
                print(f'{conllu_path} does not exist, or is not a file.')
                continue

            print(f'conllu: {conllu_path}')

            # TODO: pass in validation_info column for whether text was altered from raw form or not to add it to each conllu_by_doc dataframe
            conllu_totals_dict = describe_conllu(
                conllu_path, group_stats_dir)

            conllu_dicts.append(conllu_totals_dict)

        grp_by_conllu_df = downcast_df(pd.DataFrame(conllu_dicts))
        grp_by_conllu_df = grp_by_conllu_df.set_index('slice_name')
        write_df(grp_by_conllu_df, group_stats_dir.joinpath(f'{data_grp}_conllus.pkl.bz2'))
        # # drop doc level stats for the data group level
        # g_convert = (grp_by_conllu_df.copy()
        #              .loc[:, ~grp_by_conllu_df.columns.str.startswith('C_D')])
        # g_convert.columns = g_convert.columns.str.replace('C_', '')

        # grp_stats_dict = describe_counts(g_convert, prefix='C')
        # # * START HERE!!!! 👇
        # ## TODO: need to add in the "list object" column processing as well
        # grp_stats_dict = {f'G_{k}': v for k, v in grp_stats_dict.items()}
        grp_dict = get_upper_level_stats(grp_by_conllu_df, 'group')
        grp_dict['G_id'] = data_grp
        # TODO : come back to this... not sure this is right.
        grp_dicts.append(grp_dict)
        
    puddin_by_grp_df = downcast_df(pd.DataFrame(grp_dicts))
    end = time.perf_counter()
    # ^ At this point there is:
    #   1. a list of dataframes:
    #       1 per conllu file, with rows for every doc in that conllu file
    #   2. a dataframe for the entire group (data origin source):
    #       indexed by conllu path
    # TODO : develop code to gather descriptive stats (1) by group and (2) for the entire corpus
    print(f'With cap set to {CAP},',
          f'processing for {data_target} took ~ {round((end-start)/60, 1)} minutes')


def write_df(df, outpath):
    path_stem = outpath.name.split(".")[0]

    is_list = df.columns.str.endswith(('lemmas', 'wlens'))
    list_cols = df.columns[is_list].to_list()
    val_cols = df.columns[~is_list].to_list()
    df = df.loc[:, val_cols + list_cols]

    reduced_df = df.copy()
    for col in list_cols:
        df.loc[:, col] = (df.loc[:, col]
                          .apply(lambda obj:
                                 ' '.join(obj.astype('string'))))
        reduced_df = reduced_df.loc[:, reduced_df.columns != col]

    reduced_df.to_csv(outpath.with_name(path_stem+'-metrics-only.csv'))
    # df.to_json(outpath.with_name(path_stem+'.json'),
    #            orient="index", indent=4, double_precision=4)

    if '.pkl' in outpath.suffixes:
        df.to_pickle(outpath)
        return

    # if outpath.suffix != 'csv':
    #     print('Warning: output path is not csv or pkl. Overriding to save as csv.')
    #     outpath = outpath.with_name(outpath.stem + '.csv')

    # df.to_csv(outpath)
    return


def describe_conllu(conllu_path, group_stats_dir):
    # > Iterate over conllu file and collect word & character counts for each sentence
    print(f'Counting data in {conllu_path}...')
    doc_output_dir = group_stats_dir.joinpath('docs_by_conllu')
    if not doc_output_dir.is_dir():
        doc_output_dir.mkdir(parents=True)
    #! default sentence cap is to use global CAP value
    conll_iter = gen_sentence_info(conllu_path)
    sdf = pd.DataFrame(conll_iter)
    sdf = sdf.set_index('Sid')
    # TODO: return or save sentence info dataframe to file?

    # > Group by document and add stats at document level

    conllu_by_doc = get_stats_by_doc(sdf)
    conllu_by_doc = downcast_df(conllu_by_doc)
    # d_stats.info(null_counts=False, memory_usage='deep')

    conllu_by_doc_path = doc_output_dir.joinpath(
        f'{conllu_path.stem}_docs.pkl.bz2')
    print(f'saving conllu by doc info to {conllu_by_doc_path}...')
    write_df(conllu_by_doc, conllu_by_doc_path)

    # > calculate totals across slice/conllu file
    # // # > transpose and add feature to sort through stats
    # // Tdstats = d_stats.transpose().assign(
    # //     feature=d_stats.columns.astype('string').str.rsplit('_', 1).str.get(0))
    # // for feat, stats in Tdstats.groupby('feature'):
    # //     if '_' in feat:
    # //         print(feat)
    # //         print(stats)
    # //     else:
    # //         print('[Series object]')
    # //         print(stats.transpose().dtypes)

    c_dict = get_upper_level_stats(conllu_by_doc, 'conllu')
    c_dict.update({'conllu_path': conllu_path,
                   'slice_name': conllu_path.stem.rsplit('_', 1)[1]})

    # TODO: append conllu_dict to iter of all conllus initialized before the for-loop
    # ? or just add to existing `grp_info` loaded from file? grp_info should be df for a data group/source file with row for every slice and columns as (validation) info for each slice

    return c_dict


def gen_sentence_info(conllu_path, sentence_cap: int = CAP):

    file_iter = pyconll.iter_from_file(conllu_path)
    if sentence_cap:
        print(f'IN DEVELOPMENT: only reading first {sentence_cap} sentences')
        file_iter = islice(file_iter, sentence_cap)

    for sentence in file_iter:
        # if sentence.meta_present('newdoc id'):# and sentence.meta_value('newdoc_id') != doc_id:
        #     doc_id = sentence.meta_value('newdoc id')
        #     print(doc_id)
        # elif not doc_id:
        #     print(f'! WARNING: doc {doc_id} info not found!')
        # elif not sentence.id.startswith(doc_id):
        #     print(f'~!~ WARNING: doc {doc_id} and sentence ids do not match!')
        # # print(sentence.text)
        yield from read_sentence(sentence)


def read_sentence(sentence):
    sent_tuple = namedtuple(
        'sent_counts',
        ['Sid', 'Did', 'txt',
         'lemmas', 'words',
         # ? is `wlens` (word lengths) truly needed?
         #  ^(mean wlen can be calculated by `chr_count/wrd_count` at any level)
         'wlens',
         'wrd_count', 'chr_count',
         'wlen_mean'])
    # * NOTE: this excludes all punctuation symbols!
    tok_objects = tuple(tok for tok in sentence._tokens
                        if tok.deprel != 'punct')
    lemmas = pd.Series(tok.lemma for tok in tok_objects)
    words = pd.Series(tok.form for tok in tok_objects)
    word_lengths = pd.Series(len(word) for word in words)
    word_count = len(words)
    char_count = sum(word_lengths)
    doc_id = sentence.id.rsplit('_', 1)[0]
    yield sent_tuple(sentence.id, doc_id, sentence.text,
                     lemmas, words, word_lengths,
                     word_count, char_count,  # lemma_count == word_count
                     char_count/word_count)


def describe_counts(df, prefix: str = 's'):

    counts_desc_dict = {}
    counts = df.loc[:, df.columns.str.endswith('count')]
    # > pull out only the distinguishing str
    counts.columns = counts.columns.str.replace('count', '').str.strip('_')

    # first describe metric is "count" ~ do not need, so drop
    counts_desc = counts.describe().iloc[1:, :]

    t_counts_desc = counts_desc.transpose()
    # t_counts_desc = t_counts_desc.assign(min=t_counts_desc.loc[:, 'min'].astype('int'),
    #                                      max=t_counts_desc.loc[:, 'max'].astype('int'))
    # > keep sum values=totals separate because they get a different naming scheme
    totals = counts.sum()

    # for each row of combined descriptive stats automated df:
    #   1. pull out row as its *own* dataframe
    #   2. rename cols to indicate row/data (i.e. 'wrd_' or 'chr_')
    for row_ix in t_counts_desc.index:
        row_df = t_counts_desc.loc[[row_ix], :]
        # > row_df columns are the statistical metrics
        # > row_df rows are labels of the data being described
        feat_name = f'{prefix}ln{row_ix}_'
        row_df.columns = feat_name + row_df.columns
        row_df.loc[:, f'{row_ix}_count'] = int(totals.at[row_ix])
        row = row_df.squeeze()
        row_dict = row.to_dict()
        counts_desc_dict.update(row_dict)

    return counts_desc_dict


def describe_word_level_series(wrd_lvl_ser: pd.Series, metric_prefix: str):

    if not metric_prefix.endswith('_'):
        metric_prefix += '_'
    ser_desc = wrd_lvl_ser.describe()
    ser_desc = ser_desc.loc[ser_desc.index != 'count']
    ser_desc.index = metric_prefix + ser_desc.index
    return ser_desc.to_dict()


def generate_flat_iter(nested_iter):
    yield from (x for x_iter in nested_iter for x in x_iter)


def get_stats_by_doc(sdf: pd.DataFrame):
    """returns dataframe of descriptive statistics at the document level
        for each document in the given conllu_path. 
        Raw data included in output: 
            `D_lemmas` ~ series containing all lemmas in doc
            `D_wlens` ~ series containing all word lengths in doc
        Feature measurements/statistics: 
            `D_Swrd` ~ document sentence length in words
            `D_Schr` ~ document sentence length in characters
            `D_lmm` ~ document lemma descriptive stats
            `D_wlen` ~ document word length descriptive stats

        Args:
            sdf (pd.DataFrame): dataframe of info for every sentence in the given conllu_path. Each sentence is its own row.

        Returns:
        pd.DataFrame: dataframe with doc level info for given conllu_path. Each document is its own row.
    """
    d_dicts = []
    for doc, gdf in sdf.groupby('Did'):
        # print(doc)
        doc_dict = describe_counts(gdf, 'S')
        doc_dict = {f'D_{k}': v for k, v in doc_dict.items()}

        # TODO: 👉 use `s_wlen_list` col to calculate mean doc wlen (or not??)

        doc_lemmas = pd.Series(generate_flat_iter(gdf.lemmas))
        doc_wlens = pd.Series(generate_flat_iter(gdf.wlens))
        doc_add = {

            # document id
            'D_id': doc,

            # total sentences in doc (~ sentences/per doc)
            # (synonymous with the `count` descriptors dropped above)
            'D_snt_count': len(gdf),

            'D_lemmas': doc_lemmas,
            # same size as series column and complicates processing
            # // 'D_lemmas_concat': ' '.join(doc_lemmas),

            # REPLACED ALL OF THIS MANUAL ENTRY WITH `describe()`
            # // 'D_wlens': doc_wlens,
            # // # total char in doc / total words in doc
            # // # 'D_wlen_mean.0': doc_dict[f'D_{prefix}chr_total']/doc_dict[f'D_{prefix}wrd_total'],
            # // # NOTE: this 👆 could also be done by getting the mean of all
            # // #   the wlen elements for each value/cell of `s_wlen_list` col,
            # // #   but that is unnecessary -> returns identical result:
            # // # ^ But, if creating series of all word lengths _anyway_, this is cleaner:
            # // 'D_wlen_mean': doc_wlens.mean(),
            # // 'D_wlen_median': doc_wlens.median(),

            # ? maybe don't need to keep the literal word lengths for stats above doc level?
            'D_wlens': doc_wlens
        }

        doc_dict.update(doc_add)
        doc_dict.update(
            describe_word_level_series(doc_lemmas, 'D_lmm'))
        doc_dict.update(
            describe_word_level_series(doc_wlens, 'D_wlen'))

        d_dicts.append(doc_dict)

    d_stats = pd.DataFrame(d_dicts)
    d_stats = d_stats.assign(
        D_id=d_stats.D_id.astype('string')).set_index('D_id')

    return d_stats


def downcast_df(df):
    # print('^^ Original Info ^^')
    df = df.convert_dtypes().infer_objects()

    # df.info(memory_usage=True)
    df.columns = df.columns.str.replace(
        '50%', 'median').str.replace('freq', 'topfreq')

    # > downcast float columns
    is_float = df.dtypes.astype('string').str.startswith(('float', 'Float'))
    df.loc[:, is_float] = df.loc[:, is_float].apply(
        lambda c: pd.to_numeric(c, downcast='float'))

    # > downcast integer columns
    is_int = df.dtypes.astype('string').str.startswith(('int', 'Int'))
    df.loc[:, is_int] = df.loc[:, is_int].apply(
        lambda c: pd.to_numeric(c, downcast='integer'))

    # print('-- Downcast Info --')
    # df.info(memory_usage=True)
    return df


def get_upper_level_stats(lower_df:pd.DataFrame, level:str):
    # drop sent level values for conllu calculations
    if level == 'conllu':
        col_key = 'D_S'
        lower_pref = 'D'
        upper_pref = 'C'
    
    elif level == 'group':
        col_key = 'C_D'
        lower_pref = 'C'
        upper_pref = 'G'
        
    elif level in ('puddin', 'top'):
        col_key = 'G_C'
        lower_pref = 'G'
        upper_pref = 'P'
        
    convert_up = lower_df.copy().loc[:, ~lower_df.columns.str.startswith(col_key)]
    convert_up.columns = convert_up.columns.str.replace(f'{lower_pref}_', '')

    upper_dict = describe_counts(convert_up, lower_pref)
    upper_dict = {f'{upper_pref}_{k}': v for k, v in upper_dict.items()}
    upper_dict[f'{upper_pref}_doc_count'] = len(lower_df)

    # > process iter object columns
    objects_df = lower_df.loc[:, lower_df.dtypes == 'object']
    for obj_col_name in objects_df.columns:
        conllu_col_name = obj_col_name.replace(lower_pref, upper_pref)
        conllu_Series = pd.Series(generate_flat_iter(
            objects_df.loc[:, obj_col_name]))
        upper_dict[conllu_col_name] = conllu_Series
        desc_dict = describe_word_level_series(
            conllu_Series,
            conllu_col_name.strip('s').replace('lemma', 'lmm'))
        upper_dict.update(desc_dict)

    # TODO: see if there are any other values worth keeping from doc level stats (e.g. most freq lemma per doc?, median max sent length per doc? 🤔🤷‍♀️)
    return upper_dict


# %%
if __name__ == '__main__':
    # ^ with slurm, this can be run with array job,
    # ^     with data_code = "Pcc${SEED}"
    target_data = 'test'
    data_identifier = f'Pcc{target_data[:2].capitalize()}'

    #! testing all present data dirs
    describe_puddin()
