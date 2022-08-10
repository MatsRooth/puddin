import logging
import time
from pathlib import Path
from multiprocessing import current_process
import pandas as pd
import sys

from pull_ids_from_conll import conllu_id_iter, reconstruct_raw_iter

VALID_EXCL_DIR_NAME = 'validated'
_LOG_INDENT = 3
_MISSING_EXCL_CODE = 'missing*'
_inform = logging.info
_warn = logging.warning
_err = logging.error
_SLICE_NUM_ZFILL = 3
# _crash = logging.critical


def _format(message):

    return message.replace('\n', f'\n{" "*_LOG_INDENT}')


def dur_round(time_dur: float):
    """take float of seconds and converts to minutes if 60+, then rounds to 1 decimal if 2+ digits

    Args:
        dur (float): seconds value

    Returns:
        str: value converted and rounded with unit label of 's','m', or 'h'
    """
    unit = 's'

    if time_dur >= 60:
        time_dur = time_dur/60
        unit = 'm'

        if time_dur >= 60:
            time_dur = time_dur/60
            unit = 'h'

    if time_dur < 10:
        dur_str = f'{round(time_dur, 2):.2f}{unit}'
    else:
        dur_str = f'{round(time_dur, 1):.1f}{unit}'

    return dur_str


def _setup_logging(task):

    # ) slurm sets cwd to logs/ directory
    logpath = Path.cwd().joinpath('validation_by_group', #'testing',
                                  f'{task.split("-")[1].zfill(2)}_{time.strftime("%Y-%m-%d_%I%M%p")}.log')
    if not logpath.parent.is_dir():
        logpath.parent.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format=(' _______________________\n| %(levelname)s: %(asctime)s\n'
                '|    %(pathname)s:%(funcName)s:%(lineno)d\n| . . . . . . .\n   %(message)s'),
        datefmt="%Y-%m-%d %I:%M%p",
        filename=str(logpath))


def assess_data_group(grp: str, info: pd.DataFrame, rawdf_path: Path, data_dir: Path):

    _setup_logging(current_process().name)
    _inform(_format(
        f'VALIDATION LOG: {grp}\nPID: {current_process().pid}'))

    raw_dataframe = _prep_raw_dataframe(info, rawdf_path)

    conll_dir = _check_conll_path(info, data_dir)

    raw_dataframe = _get_parse_status(raw_dataframe, conll_dir)

    # subset of columns (no text or paths)
    data_grp_info = (raw_dataframe
                     .assign(data_group=grp)
                     .loc[:, ['conll_id', 'data_group']])

    # * EXCLUSIONS
    excl_path = data_dir.joinpath(info.exclusions_path.iloc[0])
    excl_dataframe = _load_exclusions(excl_path)

    data_grp_info = _consolidate_info(data_grp_info, excl_dataframe)

    # * Pull info for ids that are missing from both conllu file and exclusions
    if any(data_grp_info.missing
           #! prior validated exclusions didn't get all the right info added,
           #!    so this has to be rerun regardless
           # & (data_grp_info.excl_type != _MISSING_EXCL_CODE)
           ):
        mdf = raw_dataframe.loc[data_grp_info.index[data_grp_info.missing], :]
        excl_dataframe = _add_missing_to_excl(excl_dataframe, mdf)

        # ) done this way to update dtype `category`
        x_types = data_grp_info.excl_type.astype('string')
        x_types.loc[data_grp_info.missing] = _MISSING_EXCL_CODE
        data_grp_info = data_grp_info.assign(
            excl_type=x_types.astype('category'))
    else:
        _inform(f'No original texts missing from {grp} output ^_^')

    # * identify texts which were added to the exclusions set, yet successfully parsed
    # *   i.e. "false positives" for `exclude`
    excl_dataframe = _identify_invalid_excl(
        data_grp_info, excl_dataframe, grp)
    # if not prior_validation:

    # * save exclusions to `validated/` even if unchanged, to show which have been validated
    _save_validated_excl(excl_path, excl_dataframe, data_dir)

    _inform(
        _format(f"## `{grp}` data assessment complete\n"
                f"{time.strftime('%Y-%m-%d -- %I:%M%p')}\nstatus overview:"
                f"\n{'.'*18}\n"
                f"{data_grp_info.value_counts(['excl_type', 'true_fail', 'success']).to_string()}"
                f"\n{'.'*18}\n"
                f"{data_grp_info.value_counts(['slice', 'true_fail', 'success']).to_string()}")
    )

    inv_excl_df = excl_dataframe.loc[excl_dataframe.invalid, :]

    # ) save group & invalid exclusions dataframes
    # )      since there were issues concatonating them all together
    val_by_grp_path = data_dir.joinpath(
        'info', 'validation_by_group', f'{grp}-pcc_status.pkl.gz')

    if not val_by_grp_path.parent.is_dir():
        val_by_grp_path.parent.mkdir()
    inv_excl_path = val_by_grp_path.with_name(f'{grp}_invalid-excl.pkl.gz')

    this_script_mtime = Path(
        '/share/compling/projects/puddin/script/validate_data_group.py').stat().st_mtime
    # ) if file does not exist or file predates modifications to the code generating it, write to file
    if not val_by_grp_path.is_file() or val_by_grp_path.stat().st_mtime < this_script_mtime:
        data_grp_info.to_pickle(val_by_grp_path)
    if not inv_excl_path.is_file() or inv_excl_path.stat().st_mtime < this_script_mtime:
        inv_excl_df.to_pickle(inv_excl_path)

    _inform(_format(
        f'Data group `{grp}` parsing status info and invalid exclusions saved as:\n'
        f'- {val_by_grp_path}\n'
        f'     last modified: {time.ctime(val_by_grp_path.stat().st_mtime)}\n'
        f'- {inv_excl_path}\n'
        f'     last modified: {time.ctime(inv_excl_path.stat().st_mtime)}'))

    return data_grp_info, inv_excl_df


def _prep_raw_dataframe(info, rawdf_path):

    _inform(_format(
        f'Loading initial/raw dataframe:\n > {rawdf_path}\n   ...'))

    rdf = pd.read_pickle(rawdf_path).rename(
        columns={'raw': 'raw_text',
                 'text_id': 'raw_id',
                 'data_origin_fpath': 'jsonl_path',
                 'dataframe_fpath': 'final_df_path'})

    if (rdf.jsonl_path.unique()[0].parts[-2:]
            != info.origin_filepath.unique()[0].parts[-2:]):
        _warn('meta parsing info and raw dataframe '
              'have different paths for source data')
    if (rdf.final_df_path.unique()[0].parts[-2:]
            != info.final_df_path.unique()[0].parts[-2:]):
        _warn('meta parsing info and raw dataframe show '
              'different paths for final dataframe')

    return rdf


def _check_conll_path(info, data_dir):
    conll_dir = data_dir.joinpath(Path(info.conllu_path.iloc[0]).parent)

    _inform(_format(
        'directory of conllu files to be searched:\n > ' + str(conll_dir)))

    if not conll_dir.is_dir():
        _err(f'{conll_dir} directory does not exist.')
    if not list(conll_dir.glob('*conllu')):
        _err(f'No ".conllu" files found in {conll_dir}.')
    return conll_dir


def _get_parse_status(raw_df, conll_dir):
    # ) get "raw" version for each id in the conllu output file
    parsed_info_generator = conllu_id_iter(conll_dir, 'doc')
    # this will be a generator of tuples consisting of:
    #   (filename: str, total_docs_in_file:int, iter_of_all_doc_ids:generator)
    parsed_texts_df = pd.DataFrame()
    info_strings = []
    for fstem, conllu_doc_ids in parsed_info_generator:

        these_parsed_texts = pd.DataFrame(
            reconstruct_raw_iter(conllu_doc_ids)).assign(conllu_stem=fstem).astype('string')
        doc_count = these_parsed_texts.conll_id.count()

        info_strings.append((f'{time.strftime("%Y-%m-%d -- %I:%M%p")}'
                             f'\t{fstem}.conllu\ttotal docs: {doc_count}').expandtabs(4))

        these_parsed_texts = these_parsed_texts.assign(
            docs_in_conllu=doc_count)

        parsed_texts_df = pd.concat([parsed_texts_df, these_parsed_texts])

    _inform(_format('\n'.join(info_strings)))

    raw_df = (
        (raw_df.assign(row_ix=pd.to_numeric(raw_df.index, downcast='unsigned'))
               .set_index('raw_id'))
        .join(parsed_texts_df.set_index('raw_id')))

    return raw_df


def _load_exclusions(excl_path):
    val_excl_path = _get_val_excl_path(excl_path)
    loaded = False

    if val_excl_path.is_file():
        _inform(_format('Previously validated exclusions found!'))
        xdf = pd.read_pickle(val_excl_path)
        if xdf.empty:
            _err(
                'Previously validated exclusions dataframe was empty! Loading from original path.')
        else:
            loaded = True
            _inform(
                f'Loaded exclusions from {Path(*val_excl_path.parts[-3:])}')
            xdf.columns = xdf.columns.str.replace('recorded_fail', 'true_fail')

    if not loaded:
        _inform(_format(
            f'Loading exclusions dataframe:\n > {excl_path}'))
        xdf = pd.read_pickle(excl_path)
        xdf = xdf.assign(true_fail=xdf.excl_type.str.contains('fail'))
        # any rows without a `text_id` value were added after slicing
        try:
            text_id_col = xdf.text_id
        except KeyError:
            pass
        else:
            xdf = xdf.assign(row_ix=xdf.index).set_index('text_id')

        without_raw = text_id_col.isna()
        if any(without_raw):
            xdf.loc[without_raw, 'text_id'] = [
                i.raw_id for i
                in reconstruct_raw_iter(
                    xdf.slice_id.loc[without_raw])]

    xdf = (xdf
           .loc[:, ~xdf.columns.str.startswith(('pile', 'data_origin'))]
           .convert_dtypes())

    if 'slice_numstr' in xdf.columns:
        slice_numstr = xdf.loc[:, 'slice_numstr']
    elif 'slice_id' in xdf.columns:
        slice_numstr = xdf.slice_id.fillna(0).astype('string')

        slice_assigned = ~xdf.slice_id.isna()
        if any(slice_assigned):
            slice_numstr.loc[slice_assigned] = (
                xdf.slice_id.loc[slice_assigned].astype('string')
                .str.split('_').str.get(3, '')
                .str.split('.').str.get(0, '0'))

    xdf = xdf.assign(
        excl_type=xdf.excl_type.astype('string').astype('category'),
        row_ix=pd.to_numeric(xdf.row_ix, downcast='unsigned'),
        slice_numstr=slice_numstr.astype('string').str.zfill(
            _SLICE_NUM_ZFILL).astype('category')
    )
    _inform(_format(f'Exclusions loaded:\n{xdf}'))

    return xdf


def _consolidate_info(gdf, xdf):
    # ) for every text in gdf that is also in xdf, add `excl_type` and `true_fail` values
    # ) and every text *not* in xdf will have NaN values for these new columns
    try:
        xdf_addition = xdf.loc[:, ['excl_type', 'true_fail']]
    except KeyError:
        try:
            xdf_addition = xdf.loc[:, ['excl_type', 'recorded_fail']]
        except KeyError:
            xdf_addition = xdf.loc[:, ['excl_type']]
        else:
            xdf_addition.columns = ['excl_type', 'true_fail']

    gdf = gdf.join(xdf_addition)
    gdf = gdf.assign(
        success=~gdf.conll_id.isna(),
        true_fail=gdf.true_fail.fillna(False),
        slice=gdf.conll_id.str.split(".").str.get(0).astype('string'),
        excl_type=gdf.excl_type.astype('string').astype('category'),
        data_group=gdf.data_group.astype('string').astype('category'))
    # ) identify texts which were not successfully parsed, but are not in the exclusions
    # )   i.e. "false negatives" for `exclude`
    prior_miss = gdf.excl_type == _MISSING_EXCL_CODE
    # when previously run, "missing" texts will have excl_type assigned
    # the first time through, missing status = not in conllu (success == False)
    #   but also no known "reason" for exclusion (excl_type.isna())
    gdf = (gdf.assign(missing=prior_miss | (gdf.excl_type.isna() & ~gdf.success))
           .sort_index())
    return gdf


def _add_missing_to_excl(xdf, mdf):
    excl_cols = xdf.columns.to_list()

    # if something has been marked as "missing"
    #   but has a conll_id value, something has gone wrong
    if not all(mdf.conll_id.isna()):
        _err(_format(
            ('ERROR! Marked as "missing" but conll_id associated\n'
             + '  (= Text was found in final output!):\n*- '
             + '\n* '.join(mdf.index[~mdf.conll_id.isna()].to_list()))))

    # mdf.columns: ['row_ix', 'raw_text', 'pile_set_name',
    #                'pile_set_code', 'jsonl_path', 'final_df_path', 'conll_id']
    mdf = (mdf.rename(columns={'raw_text': 'text',
                               'jsonl_path': 'data_origin_fpath',
                               'final_df_path': 'dataframe_fpath'}))

    _inform(
        _format(
            f"Unaccounted for in conllu output:\n{mdf.loc[:,['data_origin_fpath','dataframe_fpath','text']]}")
        + _format(
            '\n<!> MISSING <!> -- will be added to exclusions with type = '
            f'{_MISSING_EXCL_CODE}:\n\n{mdf.reset_index().loc[:,["raw_id"]]}\n\n'
            '\n\n'.join(
                ((f'# {i}:\n{mdf.text[i][:200]}...{mdf.text[i][-200:]}')
                 .replace("\n", "\n\t> ").expandtabs(3)
                 for i in mdf.index)))
        + '\n________________\n')

    xdf = pd.concat([xdf, mdf]).loc[:, excl_cols]

    return xdf.assign(
        true_fail=xdf.true_fail.fillna(False),
        excl_type=(xdf.excl_type.astype('string')
                   .fillna(_MISSING_EXCL_CODE).astype('category'))
    )


def _identify_invalid_excl(data_grp_info, xdf, grp):
    xdf = xdf.assign(invalid=xdf.index.isin(
        data_grp_info.index[data_grp_info.success]))

    if any(xdf.invalid):
        _inform(_format(
            f'Erroneously marked {grp} exclusions:'
            f'\n{xdf.index[xdf.invalid].to_list()}'))

    else:
        _inform(f'All {grp} exclusions in are valid ^_^')

    return xdf


def _save_validated_excl(excl_path, xdf, data_dir):
    validated_excl_path = _get_val_excl_path(excl_path)
    _inform('Saving updated exclusions dataframe '
            f'to {validated_excl_path.relative_to(data_dir)}')
    xdf[[]].to_pickle(validated_excl_path)


def _get_val_excl_path(excl_path):

    return Path(excl_path.parent,
                VALID_EXCL_DIR_NAME,  # subfolder for validated files
                excl_path.name)
