# coding=utf-8
# !/home/arh234/.conda/envs/dev-sanpi/bin/python
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
import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path

import pandas as pd
from multiprocessing_logging import install_mp_handler

from pull_ids_from_conll import conllu_id_iter, reconstruct_raw_iter

_VALID_EXCL_DIR_NAME = 'validated'
_MISSING_EXCL_CODE = 'missing*'

# assign monikers for logging actions
_inform = logging.info
_debug = logging.debug
_warn = logging.warning
_err = logging.error
_crash = logging.critical
_LOG_INDENT = 3


def _parse_args():

    parser = argparse.ArgumentParser(
        description=('program to validate puddin build'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-d', '--data_dir',
        type=Path, default='/share/compling/data/puddin',
        help=('top level directory of puddin build. Should contain subdirectories for:'
              'raw, final, and sliced dataframes (default: pile_tables/), dataframes for '
              'all excluded texts for each data group (default: pile_exclusions/), '
              'parsed conllu files sorted by data source group (i.e. for each input '
              '.jsonlines file; default: Pcc[GRP].conll/), and an info/ dir '
              'containing the cleaned processing meta data.')
    )

    parser.add_argument(
        '-g', '--data_group',
        type=str, action='append', dest='data_grps',
        default=[],
        help=('option to restrict program to only run on given data groups. '
              'String should match stem of original data files. '
              'Can be used as many times as desired.')
    )

    parser.add_argument(
        '-l', '--log_level',
        choices=['debug', 'info', 'warning', 'error'],
        type=str, default='warning',
        help=('option to specify the verbosity of logging output')
    )
    log_level_defs = {'debug': logging.DEBUG,
                      'info': logging.INFO,
                      'warning': logging.WARNING,
                      'error': logging.ERROR}

    args = parser.parse_args()

    return args.data_dir, args.data_grps, log_level_defs[args.log_level]


def _main():
    _inform(f'Starting Puddin Validation of {_DATA_GRPS}...')
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

    _inform(_format_message(f'Assessment Complete:\n{all_df.describe()}'))
    all_df.to_pickle(_INFO_DIR.joinpath('all-pcc-texts_status.pkl.gz'))


def _load_meta_info():
    meta_info_path = _INFO_DIR.joinpath('completed-puddin_meta-index.pkl')
    _inform(f'Loading processing meta info from {meta_info_path} ...')
    if not meta_info_path.is_file():
        _crash(f'{meta_info_path} does not exist. Data cannot be assessed.')
        sys.exit(1)
    meta = pd.read_pickle(meta_info_path)

    if _DATA_GRPS:
        meta = (meta.loc[(meta.data_origin_group.isin(_DATA_GRPS)), :])

        if meta.empty:
            _crash(f'No processing data found for sources: {_DATA_GRPS}.')
            sys.exit(1)

    was_overwritten = meta.slice_name.duplicated(keep='last')
    if any(was_overwritten):
        owdf = (meta.loc[:, meta.columns.isin(["slice_name", "finished_at",
                                              "conllu_mtime", "conllu_path"])]
                .assign(overwritten=was_overwritten).sort_values('conllu_path'))
        slices_with_dups = owdf.slice_name[owdf.overwritten].unique()
        list_str = '  ~ ' + ' and '.join(', '.join(slices_with_dups)
                                         .rsplit(' ', 1))
        _err(f'Multiple records found for {len(slices_with_dups)} '
             f'slices:\n{list_str}\n{"."*len(list_str)}\n{owdf}')
    # apparently some `slice_name` values are not zfilled? so can't sort by slice_name
    _debug(meta)
    return meta_info_path, meta


def _assess_files(meta):
    all_groups_list = []
    bad_excl_list = []
    excl_dir = _DATA_DIR.joinpath(meta.exclusions_path.iloc[-1]).parent
    # ) not including `parents=True` bc it really should be a problem if
    # )   the indicated exclusions directory (the parent) cannot be found
    excl_dir.joinpath(_VALID_EXCL_DIR_NAME).mkdir(exist_ok=True)
    tables_dir = _DATA_DIR.joinpath(meta.final_df_path.iloc[-1]).parent

    grp_args_iter = ((grp, df, tables_dir, _DATA_DIR)
                     for grp, df in meta.groupby('data_origin_group') if not df.empty)

    input_count = len(meta.data_origin_group.unique())
    # ) set pool `processes` argument to number of _available_ cpus
    # ) OR number of files to be searched, whichever is smaller
    cpus = min(len(os.sched_getaffinity(0)), input_count)
    _inform(_format_message(
        f'processing {input_count} inputs with {cpus} CPUs...'))
    # print(f'processing {input_count} inputs with {cpus} CPUs...')
    _start = logging.time.perf_counter()

    multiprocessing.set_forkserver_preload(
        ['pull_ids_from_conll', 'logging', 'pd', 'sys', 'os', '_DATA_DIR',
         '_inform', '_warn', '_err'
         ])

    with multiprocessing.Pool(processes=cpus) as pool:

        # NOTE: if process function takes more than 1 argument:
        #       use a. snippet "expand_argument_inputs" and `imap(_unordered)`
        #       or  b. `starmap` -> does not require printing step after, but slower
        results = pool.imap_unordered(
            _star_assess_in_parallel,
            grp_args_iter
        )

        zfill_len = len(str(input_count))
        in_name_w = 10  # = data group
        #! this is required to actually get the processes to run
        run_count = 0
        for result in results:
            # ) if assessment returned None;
            # )     i.e. could not open raw dataframe file (or other failure?)
            # ) do not increase `run_count` for "non(e)-result"
            if result is None:
                continue
            group_info, excl_info, group, dur = result
            run_count += 1
            # ) add dataframe for current group (gdf) to list of group dataframes
            all_groups_list.append(group_info)
            bad_excl_list.append(excl_info)
            # _inform(f'`{group}` assessment time:\t{dur}')
            if run_count == 1:
                print(logging.time.strftime("%Y-%m-%d _ %I:%M%p"))
                print(('  task  |  time  \t'
                       f'{"data group".ljust(in_name_w)}\n'
                       f' ------ | ------ \t'
                       f'{"-"*in_name_w}').expandtabs(3))
            print((f'{str(run_count).zfill(zfill_len).center(8)}|{dur.rjust(7)} \t'
                   f'{group.ljust(in_name_w)}').expandtabs(3))

        total_inputs_processed = run_count
        # ? Is there a better way to do this? ^^ Like, some "run" or "start" or "join" method?

    _end = logging.time.perf_counter()
    total_time = _end - _start
    _inform(f'{total_inputs_processed} data groups validated in '
            f'{dur_round(total_time)}')
    _save_bad_excl_info(bad_excl_list, excl_dir)

    return all_groups_list


def _star_assess_in_parallel(args):
    return _assess_in_parallel(*args)


def _assess_in_parallel(grp, info, tables_dir, data_dir):
    global _DATA_DIR
    _DATA_DIR = data_dir
    _grp_start = logging.time.perf_counter()

    print('\n' + _format_message(
        f'> > > starting assessment of data group {grp} \n'
        f'        {logging.time.strftime("%Y-%m-%d -- %I:%M%p")}'))
    rawdf_path = _check_meta_info(grp, info, tables_dir)
    if not rawdf_path:
        return None

    grp_info, inv_excl_df = _assess_data_group(grp, info, rawdf_path)
    _grp_finish = logging.time.perf_counter()
    time_str = dur_round(_grp_finish - _grp_start)
    return grp_info, inv_excl_df, grp, time_str


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


def _check_meta_info(grp, info, tables_dir):
    '''sanity check paths in meta processing info _for each data group_'''
    # if more than 1 path shown for finalized unsliced dataframe for given data source
    if len(info.final_df_path.unique()) != 1:
        print(
            'WARNING! More than 1 path found for unsliced dataframe for %s data source', grp)
    if len(info.origin_filepath.unique()) != 1:
        print('WARNING! different paths showing for source file')

    findf_path = _DATA_DIR.joinpath(info.final_df_path.iloc[0])
    rawdf_path = Path(findf_path.parent, "raw", findf_path.name)
    if not rawdf_path.is_file():
        print('ERROR! Raw/initial dataframe not found. '
              f'Invalid path: {rawdf_path}')
        return None

    if findf_path.parent != tables_dir:
        print(_format_message(f'{grp} dataframe paths not in {tables_dir}. \n'
              f'   Is this file supposed to be included?: {findf_path}'))

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
        print(f'No original texts missing from {grp} output ^_^')

    # * identify texts which were added to the exclusions set, yet successfully parsed
    # *   i.e. "false positives" for `exclude`
    excl_dataframe = _identify_invalid_excl(data_grp_info, excl_dataframe, grp)

    # * save exclusions to `validated/` even if unchanged, to show which have been validated
    _save_validated_excl(excl_path, excl_dataframe)
    print(_format_message(f"## `{grp}` data assessment complete\n"
                          f"{logging.time.strftime('%Y-%m-%d -- %I:%M%p')}\nstatus overview:"
                          f"\n{'.'*18}\n"
                          f"{data_grp_info.value_counts(['excl_type', 'recorded_fail', 'success'])}"
                          f"\n{'.'*18}\n"
                          f"{data_grp_info.value_counts(['slice', 'recorded_fail', 'success'])}")
          )

    inv_excl_df = excl_dataframe.loc[excl_dataframe.invalid, :]
    return data_grp_info, inv_excl_df


def _prep_raw_dataframe(info, rawdf_path):

    print(_format_message(
        f'Loading initial/raw dataframe:\n > {rawdf_path}\n   ...'))

    rdf = pd.read_pickle(rawdf_path).rename(
        columns={'raw': 'raw_text',
                 'text_id': 'raw_id',
                 'data_origin_fpath': 'jsonl_path',
                 'dataframe_fpath': 'final_df_path'})

    if (rdf.jsonl_path.unique()[0].parts[-2:]
            != info.origin_filepath.unique()[0].parts[-2:]):
        print('WARNING! meta parsing info and raw dataframe '
              'have different paths for source data')
    if (rdf.final_df_path.unique()[0].parts[-2:]
            != info.final_df_path.unique()[0].parts[-2:]):
        print('WARNING! meta parsing info and raw dataframe show '
              'different paths for final dataframe')

    return rdf


def _check_conll_path(info):
    conll_dir = _DATA_DIR.joinpath(Path(info.conllu_path.iloc[0]).parent)

    print(_format_message(
        'directory of conllu files to be searched:\n > ' + str(conll_dir)))

    if not conll_dir.is_dir():
        print(f'ERROR! {conll_dir} directory does not exist.')
    if not list(conll_dir.glob('*conllu')):
        print(f'ERROR! No ".conllu" files found in {conll_dir}.')
    return conll_dir


def _get_parse_status(raw_df, conll_dir):
    # ) get "raw" version for each id in the conllu output file
    parsed_info_generator = conllu_id_iter(conll_dir, 'doc')
    # this will be a generator of tuples consisting of:
    #   (filename: str, total_docs_in_file:int, iter_of_all_doc_ids:generator)
    parsed_texts_df = pd.DataFrame()
    for fstem, conllu_doc_ids in parsed_info_generator:

        these_parsed_texts = pd.DataFrame(
            reconstruct_raw_iter(conllu_doc_ids)).assign(conllu_stem=fstem).astype('string')
        doc_count = these_parsed_texts.conll_id.count()

        print((f'{logging.time.strftime("%Y-%m-%d -- %I:%M%p")}'
              f'\t{fstem}.conllu\ttotal docs: {doc_count}').expandtabs(4))

        these_parsed_texts = these_parsed_texts.assign(
            docs_in_conllu=doc_count)

        parsed_texts_df = pd.concat([parsed_texts_df, these_parsed_texts])

    _debug(parsed_texts_df.sample(5))

    raw_df = (
        (raw_df.assign(row_ix=raw_df.index.astype('string'))
               .set_index('raw_id'))
        .join(parsed_texts_df.set_index('raw_id')))

    return raw_df


def _load_exclusions(excl_path):
    print(_format_message(f'Loading exclusions dataframe:\n > {excl_path}'))
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
        slice=gdf.conll_id.str.split(".").str.get(0).astype('string'),
        excl_type=gdf.excl_type.astype('string').astype('category'),
        data_group=gdf.data_group.astype('string').astype('category'))
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
        print(_format_message(
            ('ERROR! Marked as "missing" but conll_id associated\n'
             + '  (= Text was found in final output!):\n*- '
             + '\n* '.join(mdf.index[~mdf.conll_id.isna()].to_list()))))

        # mdf.columns: ['row_ix', 'raw_text', 'pile_set_name',
        #                'pile_set_code', 'jsonl_path', 'final_df_path', 'conll_id']
    mdf = (mdf.assign(recorded_fail=False)
           .rename(columns={'raw_text': 'text',
                            'jsonl_path': 'data_origin_fpath',
                            'final_df_path': 'dataframe_fpath'}))

    print(_format_message(f'Unaccounted for in conllu output:\n{mdf}'))
    # raw_missing = rawdf.loc[rawdf.raw_id.isin(missing_ids), :]

    # TODO: see why this comes out like this:
    # ^ possibly because it is being forwarded through multiprocessing?
    #     WARNING:root:# pcc_test_17670:
    #     > Savings Plan and Future Cost Estimation for
    #       University of Phoenix Northwest Arkansas Campus
    #     >
    #     > Savings Plan and Future Cost Estimation
    #     >
    #     > Estimated Annual Cost in 2038
    #     >
    #     > $42,640
    #     >
    #     > Monthly savings required
    #     >
    #     > ...site and the school names are the property of their respective trademark owners.
    #     >
    #     > Contact Us
    #     >
    #     > If you represent a school and believe that data presented on
    #       this website is incorrect, please contact us.

    print(
        _format_message(
            '<!> MISSING <!> -- will be added to exclusions with type = '
            + f'{_MISSING_EXCL_CODE}:\n\n{mdf.reset_index().loc[:,["raw_id"]]}\n\n'
            + '\n\n'.join(
                ((f'# {i}:\n{mdf.text[i][:200]}...{mdf.text[i][-200:]}'
                  ).replace("\n", "\n\t> ").expandtabs(3)
                 for i in mdf.index))
        ),
        end='\n________________\n')

    xdf = pd.concat([xdf, mdf]).loc[:, excl_cols]
    return xdf


def _identify_invalid_excl(data_grp_info, xdf, grp):
    xdf = xdf.assign(invalid=xdf.index.isin(
        data_grp_info.index[data_grp_info.success]))

    if any(xdf.invalid):
        print(_format_message(
            f'Erroneously marked {grp} exclusions:'
            f'\n{xdf.index[xdf.invalid].to_list()}'))

    else:
        print(f'All {grp} exclusions in are valid ^_^')

    return xdf


def _save_validated_excl(excl_path, xdf):

    validated_excl_path = Path(excl_path.parent,
                               _VALID_EXCL_DIR_NAME,  # subfolder for validated files
                               excl_path.name)
    print('Saving updated exclusions dataframe '
          f'to {validated_excl_path.relative_to(_DATA_DIR)}')
    xdf.to_pickle(validated_excl_path)


def _save_bad_excl_info(bad_excl_list, excl_path):
    if any(not df.empty for df in bad_excl_list):
        bad_excl = pd.concat(bad_excl_list)
        _warn(_format_message(
            f'Erroneous Exclusions Found!\n{bad_excl.index}'))
        bad_excl.to_json(_INFO_DIR.joinpath('excl_mistakes.json'),
                         orient='index', indent=4)
    else:
        _inform(
            # print(
            f'No "false" exclusions found in {excl_path.relative_to(_DATA_DIR)}/')


def _save_missing_info(all_df):
    if any(all_df.missing):
        all_missing_path = _INFO_DIR.joinpath('missing_texts.csv')
        log_message = _format_message(('Some original texts were unaccounted for!\n'
                                      'Missing text info for all data groups saved as '
                                       f'{all_missing_path.relative_to(_DATA_DIR)}/.json'))
        _warn(log_message)
        missing_df = all_df.loc[all_df.missing, :]
        missing_df.to_csv(all_missing_path)
        missing_df.reset_index().set_index(['data_group', 'raw_id'], append=True).to_json(
            all_missing_path.with_suffix('.json'), orient='index', indent=4)
    else:
        _inform('All original texts accounted for ^_^')


def _format_message(message):

    return message.replace('\n', f'\n{" "*_LOG_INDENT}')


if __name__ == '__main__':

    START = logging.time.perf_counter()

    _DATA_DIR, _DATA_GRPS, _log_level = _parse_args()
    _INFO_DIR = _DATA_DIR.joinpath('info')

    multiprocessing.set_start_method('forkserver')

    logging.basicConfig(
        level=_log_level,
        # format='[%(levelname)s] %(asctime)s\n  > %(message)s',
        format=' _______________________\n| %(levelname)s: %(asctime)s\n|    %(pathname)s:%(funcName)s:%(lineno)d\n| . . . . . . .\n   %(message)s',
        datefmt="%Y-%m-%d %I:%M%p",
        stream=sys.stdout,
        # todo: find a way to get the log of everything together
        # filename=str(Path('/share/compling/projects/puddin', 'logs',
        #                   f'validation{logging.time.strftime("%Y-%m-%d_%H:%M")}.log'))
    )
    install_mp_handler()
    _MP_LOGGER = multiprocessing.log_to_stderr()
    _MP_LOGGER.setLevel(logging.WARNING)

    _main()

    END = logging.time.perf_counter()
    _inform(
        f'Total assessment time: {dur_round(END - START)} seconds')
