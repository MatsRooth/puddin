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
# import argparse
from datetime import datetime
import sys
from pathlib import Path
import logging
import pandas as pd

from pull_ids_from_conll import conllu_id_iter, reconstruct_raw_iter

_DATA_DIR = Path('/share/compling/data/puddin')
_INFO_DIR = _DATA_DIR.joinpath('info')
# TODO: temp -- REMOVE
_DATA_GRPS = [
    # 'test',
    'val',
    # '29'
]
_VALID_EXCL_DIR_NAME = 'validated'
_MISSING_EXCL_CODE = 'missing*'
# _DF1_NAME = f'pile_{_DATA_GRPS[0]}_Pile-CC_df.pkl.gz'

_inform = logging.info
_debug = logging.debug
_warn = logging.warning
_err = logging.error
_crash = logging.critical

logging.basicConfig(level=logging.INFO,
                    format='\n[%(levelname)s]:\n%(message)s')


def _main():
    # ) Load meta info dataframe
    meta_info_path, meta = _load_meta_info()
    # For each row (i.e. slice) compare the text ids found in the files at
    # the following paths: raw, final, conllu. Make sure any missing
    # from conllu are in exclusions dataframe.
    all_groups_list = _assess_files(meta)

    if not all_groups_list:
        _crash('Data validation failed. Could not find files '
               f'using paths specified in {meta_info_path}.')
        sys.exit(1)
    all_df = pd.concat(all_groups_list).sort_values('conll_id')

    _save_missing_info(all_df)

    _inform(f'Assessment Complete:\n{all_df.describe()}')
    all_df.to_pickle(_INFO_DIR.joinpath('all-pcc-texts_status.pkl.gz'))


def _load_meta_info():
    meta_info_path = _INFO_DIR.joinpath('completed-puddin_meta-index.pkl')
    _inform(f'Loading processing meta info from {meta_info_path}...')
    if not meta_info_path.is_file():
        _crash(f'{meta_info_path} does not exist. Data cannot be assessed.')
        sys.exit(1)
    meta = pd.read_pickle(meta_info_path)

    # TODO: temp filter for _debugging -- REMOVE
    meta = (meta.loc[(meta.data_origin_group.isin(_DATA_GRPS)), :])

    if meta.empty:
        _crash(f'No processing data found for sources: {_DATA_GRPS}.')
        sys.exit(1)
    was_overwritten = meta.slice_name.duplicated(keep='last')
    if any(was_overwritten):
        owdf = (meta.loc[:, meta.columns.isin(["slice_name", "finished_at",
                                              "conllu_mtime", "conllu_path"])]
                .assign(overwritten=was_overwritten).sort_values('slice_name'))
        slices_with_dups = owdf.slice_name[owdf.overwritten].unique()
        list_str = '  ~ ' + ' and '.join(', '.join(slices_with_dups)
                                         .rsplit(' ', 1))
        _err(f'Multiple records found for {len(slices_with_dups)} '
             f'slices:\n{list_str}\n{"."*len(list_str)}\n{owdf}')
    #! apparently some `slice_name` values are not zfilled? throws off sorting this way
    _debug(meta.sample(min(5, len(meta))))
    _debug(meta.describe().round(1))
    return meta_info_path, meta


def _assess_files(meta):
    all_groups_list = []
    bad_excl_list = []

    excl_dir = _DATA_DIR.joinpath(meta.exclusions_path.iloc[-1]).parent
    # ) not including `parents=True` bc it really should be a problem if
    # )   the indicated exclusions directory (the parent) cannot be found
    excl_dir.joinpath(_VALID_EXCL_DIR_NAME).mkdir(exist_ok=True)

    tables_dir = _DATA_DIR.joinpath(meta.final_df_path.iloc[-1]).parent

    # TODO: PARALLIZE THIS LOOP! using multiprocessing
    for grp, info in meta.groupby('data_origin_group'):
        if info.empty:
            continue
        _inform(f'\n> > > starting assessment of `{grp}` data \n'
                f'        {datetime.now().strftime("%R -- %Y-%m-%d")}')
        rawdf_path = _check_meta_info(grp, info, tables_dir)
        if not rawdf_path:
            continue

        grp_info, inv_excl_df = _assess_data_group(grp, info, rawdf_path)
        # ) add dataframe for current group (gdf) to list of group dataframes
        all_groups_list.append(grp_info)
        bad_excl_list.append(inv_excl_df)

    _save_bad_excl_info(bad_excl_list, excl_dir)

    return all_groups_list


def _check_meta_info(grp, info, tables_dir):
    '''sanity check paths in meta processing info _for each data group_'''
    # if more than 1 path shown for finalized unsliced dataframe for given data source
    if len(info.final_df_path.unique()) != 1:
        _warn(
            'WARNING! More than 1 path found for unsliced dataframe for %s data source', grp)
    if len(info.origin_filepath.unique()) != 1:
        _warn('WARNING! different paths showing for source file')

    findf_path = _DATA_DIR.joinpath(info.final_df_path.iloc[0])
    rawdf_path = Path(findf_path.parent, "raw", findf_path.name)
    if not rawdf_path.is_file():
        _err('Raw/initial dataframe not found. '
             f'Invalid path: {rawdf_path}')
        return None

    if findf_path.parent != tables_dir:
        _warn(f'{grp} dataframe paths not in {tables_dir}. \n'
              f'   Is this file supposed to be included?: {findf_path}')

    return rawdf_path


def _assess_data_group(grp, info, rawdf_path):

    raw_dataframe = _prep_raw_dataframe(info, rawdf_path)

    conll_dir = _check_conll_path(info)

    raw_dataframe = _get_parse_status(raw_dataframe, conll_dir)

    # subset of columns (no text or paths)
    data_grp_info = (raw_dataframe
                     .assign(data_group=grp)
                     .loc[:, ['conll_id', 'data_group']])

    # * EXCLUSIONS
    excl_path = _DATA_DIR.joinpath(info.exclusions_path.iloc[0])
    excl_dataframe = _load_exclusions(excl_path)

    data_grp_info = _consolidate_info(data_grp_info, excl_dataframe)

    # * Pull info for ids that are missing from both conllu file and exclusions
    if any(data_grp_info.missing):
        mdf = raw_dataframe.loc[data_grp_info.index[data_grp_info.missing], :]
        excl_dataframe = _add_missing_to_excl(excl_dataframe, mdf)

        # ) this has to be done this way to update dtype category
        x_types = data_grp_info.excl_type.astype('string')
        x_types.loc[data_grp_info.missing] = _MISSING_EXCL_CODE
        data_grp_info = data_grp_info.assign(
            excl_type=x_types.astype('category'))
    else: 
        _inform('No original texts missing from processing output ^_^')

    # * identify texts which were added to the exclusions set, yet successfully parsed
    # *   i.e. "false positives" for `exclude`
    excl_dataframe = _identify_invalid_excl(data_grp_info, excl_dataframe)

    # * save exclusions to `validated/` even if unchanged, to show which have been validated
    _save_validated_excl(excl_path, excl_dataframe)

    _inform(f"## `{grp}` data assessment complete\n"
            f"{datetime.now().strftime('%R -- %Y-%m-%d')}\nstatus overview:"
            f"\n{'.'*18}\n"
            f"{data_grp_info.value_counts(['excl_type', 'recorded_fail', 'success'])}"
            f"\n{'.'*18}\n"
            f"{data_grp_info.value_counts(['slice', 'recorded_fail', 'success'])}")

    inv_excl_df = excl_dataframe.loc[excl_dataframe.invalid, :]
    return data_grp_info, inv_excl_df


def _prep_raw_dataframe(info, rawdf_path):
    _inform(f'Loading initial/raw dataframe:\n > {rawdf_path}\n   ...')
    rdf = pd.read_pickle(rawdf_path).rename(
        columns={'raw': 'raw_text',
                 'text_id': 'raw_id',
                 'data_origin_fpath': 'jsonl_path',
                 'dataframe_fpath': 'final_df_path'})

    if (rdf.jsonl_path.unique()[0].parts[-2:]
            != info.origin_filepath.unique()[0].parts[-2:]):
        _warn('WARNING! meta parsing info and raw dataframe '
              'have different paths for source data')
    if (rdf.final_df_path.unique()[0].parts[-2:]
            != info.final_df_path.unique()[0].parts[-2:]):
        _warn('WARNING! meta parsing info and raw dataframe show '
              'different paths for final dataframe')

    return rdf


def _check_conll_path(info):
    conll_dir = _DATA_DIR.joinpath(Path(info.conllu_path.iloc[0]).parent)
    _inform('directory of conllu files to be searched:\n > ' + str(conll_dir))
    if not conll_dir.is_dir():
        _err(f'{conll_dir} directory does not exist.')
    if not list(conll_dir.glob('*conllu')):
        _err(f'No ".conllu" files found in {conll_dir}.')
    return conll_dir


def _get_parse_status(raw_df, conll_dir):
    parsed_doc_ids = conllu_id_iter(conll_dir, 'doc')
    # ) get "raw" version for each id in the conllu output file
    parsed_texts = pd.DataFrame(
        reconstruct_raw_iter(parsed_doc_ids)).astype('string')
    _inform(parsed_texts.count())
    _debug(parsed_texts.sample(5))

    raw_df = (
        (raw_df.assign(row_ix=raw_df.index.astype('string'))
               .set_index('raw_id'))
        .join(parsed_texts.set_index('raw_id')))

    return raw_df


def _load_exclusions(excl_path):
    _inform(f'Loading exclusions dataframe:\n > {excl_path}')
    xdf = pd.read_pickle(excl_path)
    xdf = xdf.assign(recorded_fail=xdf.excl_type.str.contains('fail'))
    # any rows without a `text_id` value were added after slicing
    without_raw = xdf.text_id.isna()
    if any(without_raw):
        xdf.loc[without_raw, 'text_id'] = [
            i.raw_id for i
            in reconstruct_raw_iter(
                xdf.slice_id.loc[without_raw])]

    xdf = (xdf.assign(row_ix=xdf.index.astype('string')).set_index('text_id'))
    return xdf


def _consolidate_info(gdf, xdf):
    # ) for every text in gdf that is also in xdf, add `excl_type` and `recorded_fail` values
    # ) and every text *not* in xdf will have NaN values for these new columns
    gdf = gdf.join(xdf.loc[:, ['excl_type', 'recorded_fail']])
    gdf = gdf.assign(
        success=~gdf.conll_id.isna(),
        recorded_fail=gdf.recorded_fail.fillna(False),
        slice=gdf.conll_id.str.split(".").str.get(0),
        excl_type=gdf.excl_type.astype('category'),
        data_group=gdf.data_group.astype('category'))
    # ) identify texts which were not successfully parsed, but are not in the exclusions
    # )   i.e. "false negatives" for `exclude`
    gdf = (gdf.assign(missing=gdf.excl_type.isna() & ~gdf.success)
           .sort_index())
    return gdf


def _add_missing_to_excl(xdf, mdf):
    excl_cols = xdf.columns.to_list()

    # if something has been marked as "missing"
    #   but has a conll_id value, something has gone wrong
    if not all(mdf.conll_id.isna()):
        _err('Marked as "missing" but conll_id associated\n' +
             '  (= Text was found in final output!):\n*- ' +
             '\n* '.join(mdf.index[~mdf.conll_id.isna()].to_list()))

        # mdf.columns: ['row_ix', 'raw_text', 'pile_set_name',
        #                'pile_set_code', 'jsonl_path', 'final_df_path', 'conll_id']
    mdf = (mdf.assign(recorded_fail=False)
           .rename(columns={'raw_text': 'text',
                            'jsonl_path': 'data_origin_fpath',
                            'final_df_path': 'dataframe_fpath'}))
    _debug(mdf.head())
    # raw_missing = rawdf.loc[rawdf.raw_id.isin(missing_ids), :]
    _warn('<!> MISSING <!> -- will be added to exclusions with type = '
          f'{_MISSING_EXCL_CODE}:\n\n{mdf.reset_index().loc[:,["raw_id"]]}\n\n'
          '\n\n'.join((f'# {i}:\n{mdf.text[i][:200]}...{mdf.text[i][-200:]}'
                       .replace("\n", "\n\t> ").expandtabs(3)
                       for i in mdf.index)
                      )
          )

    xdf = pd.concat([xdf, mdf]).loc[:, excl_cols]
    return xdf


def _identify_invalid_excl(data_grp_info, xdf):
    xdf = xdf.assign(invalid=xdf.index.isin(
        data_grp_info.index[data_grp_info.success]))
    if any(xdf.invalid):
        _inform('Erroneously marked exclusions:'
                f'\n{xdf.index[xdf.invalid].to_list()}')
    else: 
        _inform('All exclusions are valid ^_^')

    return xdf


def _save_validated_excl(excl_path, xdf):

    validated_excl_path = Path(excl_path.parent,
                               _VALID_EXCL_DIR_NAME,  # subfolder for validated files
                               excl_path.name)
    _inform('Saving updated exclusions dataframe '
            f'to {validated_excl_path.relative_to(_DATA_DIR)}')
    xdf.to_pickle(validated_excl_path)


def _save_bad_excl_info(bad_excl_list, excl_path):
    if any(not df.empty for df in bad_excl_list):
        bad_excl = pd.concat(bad_excl_list)
        _warn(f'Erroneous Exclusions Found!\n{bad_excl.index}')
        bad_excl.to_json(_INFO_DIR.joinpath('excl_mistakes.json'),
                         orient='index', indent=4)
    else:
        _inform(
            f'No "false" exclusions found in {excl_path.relative_to(_DATA_DIR)}')


def _save_missing_info(all_df):
    if any(all_df.missing):
        all_missing_path = _INFO_DIR.joinpath('missing_texts.csv')
        _inform(
            f'Missing text info for all data groups saved as {all_missing_path.relative_to(_DATA_DIR)}/.json')
        missing_df = all_df.loc[all_df.missing, :]
        missing_df.to_csv(all_missing_path)
        missing_df.reset_index().set_index(['data_group', 'raw_id'], append=True).to_json(
            all_missing_path.with_suffix('.json'), orient='index', indent=4)
    else:
        _inform('All original texts accounted for ^_^')


if __name__ == '__main__':

    start = datetime.now()
    _main()
    end = datetime.now()
    _inform(
        f'Total assessment time: {round((end - start).total_seconds(),2)} seconds')
