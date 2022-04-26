# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint

import jsonlines
import pandas as pd
import stanza
from unidecode import unidecode

from pile_regex_imports import (bracket_url, code_regex, defwiki,
                                end_of_line_abbr, extra_newlines, json_regex,
                                likely_html, likely_url, linebreak_is_sent,
                                midword_punc_regex, missing_space_regex,
                                mixed_letter_digit_regex, punc_only,
                                solonew_or_dupwhite, underscore_regex, wikipat)

_DOC2CONLL_TEXT = stanza.utils.conll.CoNLL.doc2conll_text

_SCRIPT_START_TIME = datetime.now()
print(f'started: {_SCRIPT_START_TIME.ctime()}')
_SLICE_INFO_FILE_PREFIX = 'slice-info_'
_DATAFRAMES_DIRNAME = 'pile_tables'
_EXCLUSIONS_DIRNAME = 'pile_exclusions'
_SLDF_ROW_LIMIT = 9999
pd.set_option('display.max_colwidth', 80)
_UNK_CHAR_STR = '<__?UNK__>'
_PILE_SET_CODE_DICT = {'Gutenberg (PG-19)': 'PG19',
                       'Books3': 'Bks3',
                       'BookCorpus2': 'BkC2',
                       'Pile-CC': 'Pcc',
                       'OpenWebText2': 'OWT2'}
_UNWANTED_COLS = ('raw', 'index', 'level_0')
_NLP = None
_DESTINATION = None


def _main():
    args = parse_arg_inputs()

    global _SLDF_ROW_LIMIT
    _SLDF_ROW_LIMIT = args.output_size

    confirm_destination_dir(args.destination)

    input_files = [inpf for inpf in args.input_files if inpf.exists()]
    if set(args.input_files) - set(input_files):
        print('Warning: Not all inputs could be found:')
        pprint(list(set(args.input_files) - set(input_files)))

    init_df_paths = []
    init_js_paths = get_jsonl_paths(args)
    glob_expr = args.glob_expr
    # if there were input files given via the `-i` flag
    if input_files:
        init_df_paths = [p for p in input_files
                         if '.pkl' in p.suffixes]

    # won't have both jsonl and pkl.gz files in the same directory
    elif '.pkl' in glob_expr or _DATAFRAMES_DIRNAME in glob_expr:
        print('seeking dataframe files matching', glob_expr)

        # disallow exclusions data files unless exclusions
        # directory is specified in glob string
        if _EXCLUSIONS_DIRNAME in glob_expr:
            init_df_paths = [p for p in Path('/').glob(glob_expr)
                             if p.is_file() and '.pkl' in p.suffixes]

        else:
            init_df_paths = [p for p in Path('/').glob(glob_expr)
                             if p.is_file() and '.pkl' in p.suffixes
                             and _EXCLUSIONS_DIRNAME not in p.parts]

    data_selection = init_js_paths + init_df_paths
    if not data_selection:
        sys.exit('No data selected. Exiting.')

    print('Initial data selection:')
    pprint([get_print_path(path) for path in data_selection])
    js_paths = init_js_paths
    df_paths = init_df_paths
    if data_selection and not args.Reprocess:
        js_paths, df_paths = check_processing_status(args, data_selection)

    if not df_paths + js_paths:
        sys.exit('No valid files in need of processing. Exiting.')

    # initiate language model for dependency parsing (load just once)
    # * Note:
    # * standfordNLP does not have multi-word token (mwt) expansion
    # * for English, so `mwt` processor is not required for dependency parsing
    # (https://github.com/stanfordnlp/stanza/issues/297#issuecomment-627673245)
    global _NLP
    try:
        print('Loading dependency parsing pipeline...')
        _NLP = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse')
    except stanza.pipeline.core.ResourcesFileNotFoundError:
        print('Language model not found. Downloading...')
        stanza.download('en')
        print('Loading dependency parsing pipeline...\n')
        _NLP = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse')

    if js_paths:
        subcorpus_str = args.corpus_selection
        print('\nsubcorpora selection:', subcorpus_str)

    step_count = 1
    if df_paths:
        print('Dataframes to be processed:')
        pprint([str(path) for path in df_paths])
        slice_paths = [p for p in df_paths if 'slices' in p.parts]
        fulldf_files = list(set(df_paths) - set(slice_paths))

        if fulldf_files:

            print(f'\n\n*** ({step_count}) Unsliced Dataframe(s) ***')
            step_count += 1
            full_dfs_gen = process_pickledf(fulldf_files, data_selection)
            for df in full_dfs_gen:
                df = pop_unwanted_cols(df)
            slice_df(df)

        # TODO : move this up to parse before the full dataframes
        if slice_paths:
            print(f'\n\n*** ({step_count}) Previously Sliced Dataframes ***')
            step_count += 1

        for slice_path in slice_paths:

            print('Processing dataframe slice',
                  get_print_path(slice_path))
            # look for slice meta info in tmp slices dir for data group
            slices_metadf_search = slice_path.parent.rglob(
                get_metadf_fname(slice_path.stem
                                 .split('_')[1]  # get second _ delimited chunk
                                 .rsplit('-', 1)[0])  # pop off the slice number
            )
            try:
                slice_info_path = most_recent(slices_metadf_search)
            except ValueError:
                print('Error: script no longer supports processing sliced dataframe '
                      'without corresponding slice index dataframe.\n'
                      'Skipping file. To process this data, run from corresponding final dataframe '
                      f'in ./{_DATAFRAMES_DIRNAME}/')
                continue
            else:
                slices_metadf = pd.read_csv(slice_info_path, index_col=0)

                # (only for file saving/loading purposes)
                slices_metadf.index.name = 'slice_number'

            sldf = pd.read_pickle(slice_path)
            sldf = pop_unwanted_cols(sldf)

            process_slice(sldf, slices_metadf)

    if js_paths:
        print(
            f'\n\n*** ({step_count}) The Pile\'s Original Data Files (.jsonl) ***')
        for df in process_raw_jsonlines(js_paths, subcorpus_str):
            df = pop_unwanted_cols(df)
            slice_df(df)


def confirm_destination_dir(dest_dir):
    global _DESTINATION
    _DESTINATION = dest_dir.joinpath('puddin')
    write_access = False
    if not _DESTINATION.is_dir():
        try:
            _DESTINATION.mkdir()
        except OSError:
            pass
        else:
            write_access = True
    else:
        try:
            _DESTINATION.joinpath('write_access.txt').write_text(
                f'Write Access to {_DESTINATION} confirmed.')
        except OSError:
            pass
        else:
            write_access = True
    if not write_access:
        sys.exit(f'!! Destination directory {_DESTINATION} is not writable. '
                 'Please specify a different location for destination of output files.')

    print('All output will be saved to', _DESTINATION)


def pop_unwanted_cols(df: pd.DataFrame,
                      discard_col_names=_UNWANTED_COLS):
    for col_name in discard_col_names:
        if col_name in df.columns:
            df.pop(col_name)
    return df


def get_jsonl_paths(args):
    inputs = args.input_files
    glob_expr = args.glob_expr
    print('+ raw jsonl files selected to process:')
    jsonl_list = None

    if inputs:
        jsonl_list = [i for i in inputs if i.suffix == '.jsonl']

    else:
        print('seeking jsonl files:', glob_expr)
        jsonl_list = [p for p in Path('/').glob(glob_expr)
                      if p.is_file() and p.suffix == '.jsonl']

        if not jsonl_list and glob_expr.endswith('jsonl'):
            sys.exit('Error: No input files or valid search '
                     '(glob) expression specified. See --help for more info.')

    return jsonl_list


def check_processing_status(args, data_selection):
    js_paths = []
    df_paths = []
    print('seeking existing progress on selected files...'
          '\n---------------')
    for datapath in data_selection:
        if any(part.startswith('.') for part in datapath.parts):
            print(f'\n > Ignoring path in hidden dir, {datapath}')
            continue
        fname = datapath.name
        print('\n', datapath)
        if not datapath.is_file():
            print(' x - does not exist! Skipping.')
            continue

        datadir = datapath.parent
        path_mod_time = datapath.stat().st_mtime
        data_file_stem = datapath.stem
        is_df = '.pkl' in datapath.suffixes
        is_js = datapath.suffix == '.jsonl'
        is_slice = 'slices' in datapath.parts
        pile_set_name = (args.corpus_selection if is_js
                         else data_file_stem.split('_')[2])
        data_group = (data_file_stem if is_js
                      else data_file_stem.split('_')[1])

        if is_df and is_slice:
            finished_slice = (datadir.parent.joinpath(fname)
                              if datadir.name == 'tmp' else datapath)
            # TODO : add warning or input request here if slice's info csv cannot be located
            if (finished_slice.exists()
                    and finished_slice.stat().st_mtime >= path_mod_time):

                has_conllu = confirm_conllu(finished_slice)
                if not has_conllu:
                    df_paths.append(finished_slice)
                    if finished_slice != datapath:
                        print(' -> final slice df file:',
                              get_print_path(finished_slice))
                    else:
                        print(' -> No prior processing found.')

            else:
                df_paths.append(datapath)
                print(' -> No prior processing found.')

            continue

        # * this is necesary for any entered dataframe pkl files not prefixed with `pile_`;
        # *   e.g. pile_tables/supposed-fails_05_Pile-CC_df.pkl.gz
        elif is_df:
            finalfull_dfpath = get_dfpkl_outpath(data_file_stem)
        else:
            finalfull_dfpath = get_dfpkl_outpath(data_group,
                                                 pile_set_name)
        toplevel_dfdir = finalfull_dfpath.parent
        fulldf_filename = finalfull_dfpath.name
        if fulldf_filename in (f.name for f in df_paths):
            print(' x - Data already included '
                  'in most advanced path selection. Skipping.')
            continue

        # seek pre-sliced dataframes for given data if "reslice" option not selected
        # *note: leaving this before determining if there is a final df because slice files could
        # *  theoretically exist independent of the full dataframe
        # *  (i.e. if they have been moved from where they were originally created)
        if not args.reSlice:
            # TODO : save slice info to "finished" slice dir, above `tmp/`
            slices_info_glob = toplevel_dfdir.glob(
                'slices/**/' + get_metadf_fname(data_group))
            try:
                newest_sliceinfo_path = most_recent(slices_info_glob)
            except ValueError:
                pass
            else:
                # TODO: use file path info now in slice-index file
                #   (tmp slices, finished slices, and corresponding conllu paths)
                slice_info = pd.read_csv(newest_sliceinfo_path)

                parent_dir = newest_sliceinfo_path.parent
                slice_dir = parent_dir.parent if parent_dir.name == 'tmp' else parent_dir
                slice_fname_pattern = f'pile_{data_group}-*.pkl*'
                complete = []
                incomplete = []
                final_slices = list(slice_dir.glob(slice_fname_pattern))
                for f in final_slices:
                    if confirm_conllu(f):
                        complete.append(f)
                    else:
                        incomplete.append(f)
                tmp_slices = [
                    path for path
                    in slice_dir.joinpath('tmp').glob(slice_fname_pattern)
                    if path.name not in (f.name for f in final_slices)
                ]
                incomplete += tmp_slices
                # *if slices exist
                if incomplete:
                    df_paths += incomplete
                    print(' -> df slices:')
                    for f in incomplete:
                        print_path = get_print_path(f)
                        print('  +', print_path)
                    continue
                if complete:
                    print(' + All slices have been fully processed into conllu files.')
                    continue

        # * if reSlice *is* true, unlink all related files
        # * (i.e. delete them/clear the directories)
        #! slice dir naming was redone so that each single "data_group" (original jsonl file) has its own dir
        else:

            slice_dir_path = get_dfpkl_outpath(
                finalfull_dfpath.stem, slice_id='_').parent
            slice_dir_path = slice_dir_path.relative_to(
                Path.cwd()) if Path.cwd().name in slice_dir_path.parts else slice_dir_path
            if slice_dir_path.is_dir():
                print(
                    '   * reSlice requested\n   >> clearing all previous files from', slice_dir_path)
                for path_obj in slice_dir_path.rglob('*'):
                    if path_obj.is_file():
                        path_obj.unlink()

        # * this was `Path.cwd().glob...` but change comes with update to arg structure.
        # * This means it is not completely backwards compatible
        #  bc files saved to different location will not be located.
        #  --> good idea to manually move the previously processed files you want to keep
        matching_fulldfs = tuple(_DESTINATION.glob(
            f'{_DATAFRAMES_DIRNAME}/**/{fulldf_filename}'))

        if not matching_fulldfs and is_js:
            js_paths.append(datapath)
            print(' -> No prior processing found.')
            continue

        final_df_list = [p for p in matching_fulldfs
                         if not {'tmp', 'raw'}.intersection(set(p.parts))]

        if final_df_list:
            final_df_path = most_recent(final_df_list)
            if final_df_path.stat().st_mtime >= path_mod_time:
                df_paths.append(final_df_path)
                if final_df_path != datapath:
                    print(' -> final full df:\n  +',
                          get_print_path(final_df_path))
                else:
                    print(' -> No prior processing found.')
        else:
            tmp_df_list = [p for p in matching_fulldfs
                           if 'tmp' in p.parts]
            if tmp_df_list:

                tmp_df_path = most_recent(tmp_df_list)
                if tmp_df_path.stat().st_mtime >= path_mod_time:
                    df_paths.append(tmp_df_path)
                    if tmp_df_path != datapath:
                        print(' -> partially processed df:\n  +',
                              get_print_path(tmp_df_path))
                    else:
                        print(' -> No prior processing found.')

            else:
                raw_df_list = [p for p in matching_fulldfs
                               if 'raw' in p.parts]
                try:
                    raw_df_path = most_recent(raw_df_list)
                except ValueError:

                    print('Error in file selection. No final, tmp, or raw dataframe '
                          'files found in paths matching dataframe paths:')
                    pprint([get_print_path(m) for m in matching_fulldfs])

                else:
                    if raw_df_path.stat().st_mtime >= path_mod_time:
                        df_paths.append(raw_df_path)
                        if raw_df_path != datapath:
                            print(' -> raw df:\n  +',
                                  get_print_path(raw_df_path))
                        else:
                            print(' -> No prior processing found.')

    # remove duplicates
    if df_paths and any(pd.Series(df_paths).value_counts() > 1):
        print(pd.Series(df_paths).value_counts())
        print('Removing data duplicates...')
        df_paths = list(set(df_paths))
    if js_paths and any(pd.Series(js_paths).value_counts() > 1):
        print(pd.Series(js_paths).value_counts())
        print('Removing data duplicates...')
        js_paths = list(set(js_paths))
    print('-----------------------------------------\n\n==> Data Selection:')
    pprint([get_print_path(p) for p in js_paths + df_paths])
    return js_paths, df_paths


def get_print_path(f):
    try:
        print_path = f.relative_to(_DESTINATION.parent)
    except ValueError:
        print_path = f.relative_to(f.parent.parent.parent)
    return str(print_path)


def get_metadf_fname(data_group):
    return f'{_SLICE_INFO_FILE_PREFIX}{data_group}.csv'


def confirm_conllu(final_slice: Path):
    pile_set_name = final_slice.stem.split('_')[2]
    data_group = final_slice.stem.split('_')[1]
    data_group, slice_num = data_group.rsplit('-', 1)
    path_mod_time = final_slice.stat().st_mtime
    has_conllu = False
    conllu_path = get_conllu_outpath(data_group, slice_num,
                                     pile_set_name)
    #! final slice files are saved *after* conllu is finished
    if (conllu_path.is_file()
            and conllu_path.stat().st_mtime < path_mod_time):

        print(' ! already has corresponding conllu file',
              get_print_path(conllu_path),
              '\n  = processing complete. Skipping.')
        has_conllu = True
    return has_conllu


def most_recent(Path_iter):
    """returns the most recently modified path from an iterable of PosixPaths

    Args:
        Path_iter (iterable): iterable of PosixPath objects

    Returns:
        PosixPath: most recently modified path
    """
    return max(
        (path.stat().st_mtime, path) for path in Path_iter)[1]


def get_conllu_outpath(source_fname: str, slice_numstr: str, subset_label: str):
    '''returns path for final conllu output files'''
    # ensure the *code* (e.g. Pcc) is used, not the *name*
    subset = _PILE_SET_CODE_DICT.get(
        subset_label, subset_label.capitalize())
    out_fname = f'{subset.lower()}_eng_{source_fname}-{slice_numstr.zfill(2)}.conllu'

    # create separate conll output dir for every original source file,
    # since it looks to be gigantic
    #   Andrea Hummel on Mar 15, 2022 at 1:37 PM
    #   Redid naming schema (yesterday) so that each source jsonl file will have its
    #   own output directory (both as a `*.conll/` dir, but also in `slices/` dir)
    try:
        conlldir_num = int(source_fname)
    except ValueError:
        conlldir_id = ''.join(x[:2].capitalize()
                              for x in (source_fname.split('-')))
    else:
        conlldir_id = str(conlldir_num).zfill(2)
    out_dir = _DESTINATION.joinpath(f'{subset}{conlldir_id}.conll')

    if not out_dir.is_dir():
        out_dir.mkdir()

    return out_dir.joinpath(out_fname)


### raw processing functions ###
def process_raw_jsonlines(rfiles, subcorpus_name: str):
    for rawfile_path in rfiles:
        print(f'\n---\n\nPreprocessing {rawfile_path}...')

        df = preprocess_pile_texts(rawfile_path, subcorpus_name)

        yield df


def preprocess_pile_texts(raw_fpath: Path, selected_subset: str):

    # pile_data_path = Path('test.jsonl')
    data_source_label = raw_fpath.stem
    # path to save final version of df
    finaldf_fpath = get_dfpkl_outpath(data_source_label, selected_subset)
    # get temporary version of path for unfinished df files
    tmpdf_fpath = get_dfpkl_outpath(finaldf_fpath.stem, is_tmp=True)
    # raw path set for just this method: dataframes at any stage of pre-processing
    rawdf_dir = tmpdf_fpath.parent.parent.joinpath('raw')
    if not rawdf_dir.is_dir():
        rawdf_dir.mkdir()
    rawdf_fpath = rawdf_dir.joinpath(tmpdf_fpath.name)

    # define namedtuple to simplify dataframe creation from json object
    # //text_info = namedtuple(
    # //    'Text', ['raw', 'pile_set_name', 'pile_set_code', 'data_origin_fpath'])
    # // selected_subset = subcorpora_list
    # // if len(subcorpora_list) > 1:
    # //     print('Warning: functionality of processing 2 subcorpora '
    # //           'simultaneously is no longer supported. Only the first '
    # //           f'entry will be processed: {selected_subset}.'
    # //           f'Run the script again to process {subcorpora_list[1:]}')
    # //selection_code = global_subset_abbr_dict[selected_subset]
    # Load the (sample) jsonlines formatted (`.jsonl`) file using `jsonlines`.
    # Create a generator object which directly filters out texts from unwanted data sets.
    # Use pandas to create a flattened dataframe from the generator.
    print('  creating `jsonlines` generator for corpus selection...')
    read_t0 = datetime.now().timestamp()
    with raw_fpath.open(encoding='utf-8-sig', mode='r') as jlf:
        jlreader = jsonlines.Reader(jlf)
        jlines = jlreader.iter()
        texts = (d['text'] for d in jlines
                 if d['meta']['pile_set_name'] == selected_subset)
        read_t1 = datetime.now().timestamp()
        print(
            f'  ~ {round(read_t1 - read_t0, 3)}  sec elapsed')
        print('  building dataframe from `jsonlines` generator object...')
        #! This has to be done before the file is closed:
        #   Since we're using a generator to speed things up, the data is not fully
        #   loaded into the workspace until it's put into the dataframe.
        toDf_t0 = datetime.now().timestamp()
        df = pd.DataFrame(texts, columns=['raw'], dtype='string')

        toDF_t1 = datetime.now().timestamp()
        print(
            f'  ~ {round(toDF_t1 - toDf_t0, 3)}  sec elapsed')
    print('  ~ total time converting jsonl to dataframe:',
          timedelta(seconds=round(toDF_t1 - read_t0)))

    # Clean it up a bit, and remove duplicate text items
    df = df.drop_duplicates(subset='raw').reset_index(drop=True)
    df = (df.assign(pile_set_name=selected_subset,
                    pile_set_code=_PILE_SET_CODE_DICT[selected_subset])
          .astype(dtype={'pile_set_name': 'category',
                         'pile_set_code': 'category'})
          )
    # //df = df.assign(pile_set_name=df.pile_set_name.astype('category'))

    # Create codes for data subsets
    # (code dict is now a global variable)
    # //codes = (global_subset_abbr_dict[n] for n in df.pile_set_name)
    # //df = df.assign(pile_set_code=pd.Categorical(codes))

    #! Since the script cannot currently distinguish between
    # a partial and complete `raw` dataframe, no intermediate
    # saves should be used since they would only introduce errors
    # //# save tmp df
    # // df.to_pickle(rawdfpath)

    print('  adding subset codes & text IDs...')
    codedf = create_ids(df, data_source_label=data_source_label)
    # // df = codedf[['text_id', 'raw', 'pile_set_name', 'pile_set_code']]
    # // df = codedf
    first_cols = ['text_id', 'raw']
    # //cols = [first_cols + codedf.columns.loc[~codedfcolumns.isin(first_cols)]]]
    following_cols = list(set(codedf.columns) - set(first_cols))
    df = codedf[first_cols + following_cols]
    df = (df.assign(data_origin_fpath=raw_fpath,
                    dataframe_fpath=finaldf_fpath)
          .astype(dtype={'text_id': 'string',
                         'data_origin_fpath': 'category',
                         'dataframe_fpath': 'category'}))
    # //df = df.assign(data_origin_fpath=raw_file_path,
    # //               dataframe_fpath=df_output_path)

    # df = df.assign(data_origin_fpath=df.data_origin_fpath.astype('category'))
    print('time to make raw dataframe from jsonl =', timedelta(
        seconds=round(datetime.now().timestamp() - read_t0)), '\nsaving...')

    df.to_pickle(rawdf_fpath)
    print(f'raw dataframe saved to {get_print_path(rawdf_fpath)}')

    df = clean_df(df, tmpdf_fpath)

    # // print('\ndataframe info:')
    # // print(df.info())
    # // print('...')
    print('\nsaving final dataframe...')
    df.to_pickle(finaldf_fpath)
    print('Finished preprocessing and saved to', finaldf_fpath)

    return df


def get_dfpkl_outpath(stem: str,
                      subcorpus_label='',
                      slice_id=None,
                      is_tmp: bool = False,
                      is_excl: bool = False):
    """return path object for dataframes to be saved as `.pkl.gz`:
    file name template = `pile_[jsonl stem]_[subcorpus]_{df, excl}.pkl.gz`
    + If input is bare or only prefixed (`pile_[jsonl stem]`), 
        `corpus_selection` must be provided
    + all full dataframe file names/stems have 3 underscores
    Subdirectories of destination dir, Puddin/, are as follows:
        - for pre-processing saves,
            `_DATAFRAMES_DIRNAME/raw/`
        - for unfinalized saves, 
            `_DATAFRAMES_DIRNAME/tmp/`
        - for df slices before parsing is complete,
            `_DATAFRAMES_DIRNAME/slices/tmp/`
        - for *actual* dataframe slices (aligned with conllu output),
            `_DATAFRAMES_DIRNAME/slices/`
        - for exclusions dataframe, 
            `pile_exclusions/`

    Args:
        stem (str): _description_
        subcorpus_label (str, optional): _description_. Defaults to ''.
        slice_id (_type_, optional): _description_. Defaults to None.
        is_tmp (bool, optional): _description_. Defaults to False.
        is_excl (bool, optional): _description_. Defaults to False.

    Returns:
        PosixPath: path to save input dataframe to
    """

    data_type = 'excl' if is_excl else 'df'
    # to remove any '.pkl' strings still included with [path].stem attributes
    if '.' in set(stem):
        stem = stem.split('.', 1)[0]

    is_bare = False if '_' in stem else True
    is_prefixed = bool(stem.count('_')) or stem.startswith('pile_')
    is_full = stem.count('_') == 3 and is_prefixed

    df_output_dir = (_DESTINATION.joinpath(_EXCLUSIONS_DIRNAME) if is_excl
                     else _DESTINATION.joinpath(_DATAFRAMES_DIRNAME))

    # just the jsonl stem, e.g. "00", "val", etc.
    if is_bare:
        # process orig name parts and prefix 'pile_'
        source = stem
        subcorpus_label = ("-".join(subcorpus_label).replace(" ", "")
                           if isinstance(subcorpus_label, list)
                           else subcorpus_label)
        stem = f'pile_{stem}'

    # entire stem of previously created pkl output path;
    # e.g. "pile_00_Pile-CC_df" (".pkl" removed above)
    elif is_full:
        # '.pkl' will not have been included in ext but removed above
        # data_type not inherited from stem in this case because
        #   exclusions input stem might include df if stem Path attribute
        # ['pile_00', 'Pile-CC']
        stem, subcorpus_label = stem.rsplit('_', 2)[:2]
        source = stem.split('_')[1]

    # if not bare and not full, use `stem` and `subcorpus_label` as is;
    # e.g. "pile_00", "pile_val", etc.

    if subcorpus_label.lower() in [v.lower() for v in _PILE_SET_CODE_DICT.values()]:
        subcorpus_label = [k
                           for k, v in _PILE_SET_CODE_DICT.items()
                           if v == subcorpus_label][0]

    elif not subcorpus_label:
        print('WARNING: subcorpus label not provided for output path. '
              'Default "Pile-CC" inserted.')
        subcorpus_label = 'Pile-CC'

    # If path is for dataframe slice
    #!Note: need to test for None because `if slice_num` is False when 0
    # This precedes the tmp clause so that tmp slices
    #   are in `slices/tmp/` instead of `tmp/slices/`
    if slice_id is not None:
        slice_dir = df_output_dir.joinpath('slices')
        #! this must precede adding the slice number to the string
        #   want, e.g., `slices/Pcc00/` not `slices/Pcc00-01/`, etc.
        dirname = get_conllu_outpath(source, str(slice_id),
                                     subcorpus_label).parent.name
        dirname = dirname.replace('.conll', '')
        df_output_dir = slice_dir.joinpath(dirname)

        stem += f'-{slice_id}'

    # if tmp save (in case of script crash)
    if is_tmp:
        df_output_dir = df_output_dir.joinpath('tmp')

    if not df_output_dir.is_dir():
        df_output_dir.mkdir(parents=True)

    return df_output_dir.joinpath(
        f'{stem}_{subcorpus_label}_{data_type}.pkl.gz')


def create_ids(df: pd.DataFrame, data_source_label: str = None, zfilled_slice_num: str = None):
    '''Create text ids from raw file name, pile subset code, and dataframe index.'''

    # //codedf = pd.DataFrame()
    codes_t0 = time.perf_counter()
    # //fullix = bool(data_source_label)
    # //sliceix = bool(zfilled_slice_num)
    # // subdf = df.loc[df.pile_set_code == code, :].reset_index()
    code = df.pile_set_code.iloc[0].lower()
    prefix = ''

    if not zfilled_slice_num:
        print('  assigning text_id for each text (i.e. row)')
        prefix = f'{code}_{data_source_label}_'
    else:
        print('  updating text_id column for slice', zfilled_slice_num)
        df = df.assign(orig_text_id=df.text_id)
        file_label = df.orig_text_id.iloc[0].split(
            '_')[1]
        # e.g. pcc_eng_00_01.
        prefix = f'{code}_eng_{file_label}_{zfilled_slice_num}.'

    # start at 1 instead of 0
    idnums = df.index + 1
    zfill_len = len(str(df.index.max()))
    idnums = idnums.astype('string').str.zfill(zfill_len)

    # e.g. pcc_val_00001; pcc_eng_00_01.0001
    df = df.assign(id_stem=prefix + idnums)

    if zfilled_slice_num:
        df = df.assign(
            # e.g. pcc_eng_00_01.0001_x000003 (indices may not match)
            text_id=(df.id_stem + '_x'
                     + df.text_id.str.rsplit('_', 1).str.get(1)))
    else:
        # e.g. pcc_val_00001
        df = df.assign(text_id=df.id_stem)

    df.pop('id_stem')
    codes_t1 = time.perf_counter()
    print(f'   ~ {round(codes_t1 - codes_t0, 3)}  sec elapsed')

    return df


# process from pickle
def process_pickledf(dfiles, init_data_paths):

    for dfpath in dfiles:
        loadstart = time.perf_counter()

        print(f'\n---\n\n## Finishing processing {get_print_path(dfpath)}'
              '\n-> Loading dataframe from compressed pickle...')

        try:
            df = pd.read_pickle(dfpath)
        except zlib.error:
            print('Error: File cannot be decompressed (zlib). Skipping.')
            continue

        # previously saved df filestems
        else:
            loadcomplete = time.perf_counter()
            print('    load time =',
                  str(timedelta(seconds=loadcomplete - loadstart))[:11],)

        # #   = pile_[original jsonl file/data source file stem]_...
        # # so index 1 gives the data_source_label when split by _
        # // data_source_label = dfpath.stem.split('_')[1]
        df_has_data_origin = 'data_origin_fpath' in df.columns
        changed = False
        if (not df_has_data_origin
                or not Path(df.data_origin_fpath.iloc[0]).is_file()):

            origin_fpath = None

            # reconstruct filename for original jsonl data file
            if df_has_data_origin:
                jsonl_fname = df.data_origin_fpath.iloc[0].name
            else:
                jsonl_fname = dfpath.stem.split('_')[1]+'.jsonl'

            # attempt to locate file in plausible directories
            init_data_dirs = list(set(p.resolve().parent
                                      for p in init_data_paths))
            possible_paths = (
                init_data_dirs
                + [_DESTINATION, Path.cwd(), Path('pile_data'),
                   Path('/data/pile')]
                + list(Path().glob('pile*')))
            possible_dirs = [d for d in possible_paths if d.is_dir()]
            i = 0
            while not origin_fpath and i < len(possible_dirs):
                data_dir = possible_dirs[i]
                origin_fpath = find_jsonl(data_dir, jsonl_fname)
                i += 1

            # if none of those worked, just put in a relative path
            if not (not origin_fpath and df_has_data_origin):

                if not origin_fpath:
                    #! Do not resolve this path! It would just be wrong, so leave it unresolved.
                    origin_fpath = Path(jsonl_fname)
                    print('** Warning: Absolute path to original data is unknown. '
                          f'Filename only path inserted instead: {str(origin_fpath)}')
                else:
                    changed = True

                df = df.assign(data_origin_fpath=origin_fpath)

        if 'dataframe_fpath' not in df.columns:
            changed = True
            df = df.assign(dataframe_fpath=dfpath)

        if changed:
            print('added data path info columns saving...')
            df.to_pickle(dfpath)
            df.loc[:, ['dataframe_fpath', 'data_origin_fpath']] = (
                df.loc[:, ['dataframe_fpath', 'data_origin_fpath']]
                .astype('category'))

        # run clean up on any dataframes in `tmp/` or `raw/`
        print('finalized dataframe?')
        if dfpath.parent.name in ('tmp', 'raw'):
            print(f'  no -> Cleaning {dfpath.name}...')
            tmpdfpath = (get_dfpkl_outpath(dfpath.stem, is_tmp=True)
                         if dfpath.parent.name == 'raw'
                         else dfpath)
            df = clean_df(df, tmpdfpath)
            print('saving finalized dataframe...')
            df.to_pickle(get_dfpkl_outpath(dfpath.stem))

        else:
            print('  yes')

        yield df


def find_jsonl(dirpath: Path, jsonl_fname: str):
    matches = list(dirpath.rglob(jsonl_fname))
    if matches:
        return most_recent(matches)
    else:
        return None


# process dataframes
def clean_df(orig_df, tmp_save_path):

    print('\nCleaning text in dataframe...')
    if any(orig_df.text_id.str.startswith(('PiCC', 'Pcc'))):

        orig_df = orig_df.assign(
            text_id=orig_df.text_id.str.replace('PiCC', 'pcc')
            .str.lower().astype('category'))

    # if it doesn't have `text` column it should have `raw` column
    #   i.e. should be from the `raw/` directory
    if 'text' not in orig_df.columns:
        orig_df = orig_df.assign(text=orig_df.raw)

    # moved this here from preprocessing method because translating encoding belongs in cleanup
    # doing it before `pull_exclusions()` because texts with errors will be excluded
    print('  translating encoding...')
    unidecode_t0 = datetime.now()
    df = orig_df.assign(
        text=orig_df.text.apply(
            lambda t: unidecode(t, errors='replace',
                                replace_str=_UNK_CHAR_STR))
    )
    unidecode_t1 = datetime.now()
    elapsed_time = get_elapsed_time(unidecode_t0, unidecode_t1)
    print(f'  ~ {elapsed_time} elapsed')

    # removing urls before pulling exclusions so that "variable" and "id" patterns
    #   will not throw out texts simply due to urls that would have been removed
    print('  removing URLs...')
    t0 = time.perf_counter()
    df = df.assign(text=df.text.apply(
        lambda t: bracket_url.sub(r'\1', t)))
    df = df.assign(text=df.text.apply(lambda t: likely_url.sub(r' ', t)))
    t1 = time.perf_counter()
    print(f'  ~ {round(t1 - t0, 2)}  sec elapsed')

    t0 = time.perf_counter()
    print('  adding missing spaces after word-edge punctuation...')
    df = df.assign(text=df.text.apply(
        lambda t: missing_space_regex.sub(r'\1\3 \2\4', t)))
    t1 = time.perf_counter()
    print(f'  ~ {round(t1 - t0, 2)}  sec elapsed')

    print('saving...')
    df.to_pickle(tmp_save_path)
    print(f'dataframe saved to {get_print_path(tmp_save_path)}')

    print('+ Excluding messy data...')
    excl_save_path = get_dfpkl_outpath(tmp_save_path.stem, is_excl=True)
    df, __ = pull_exclusions(df, excl_save_path)

    # removed this because, if the script crashes before the next save,
    # it will still run through this method regardless;
    # the exlusions have already been saved, and this takes a long time.
    # // print('saving dataframe...')
    # // df.to_pickle(tmp_save_path)

    df = df.assign(text=df.text.astype('string'))

    # clean up internet syntax quirks
    print('+ Cleaning up text...\n   - punctuation delineated text breaks')
    df = df.assign(text=df.text.apply(
        lambda t: punc_only.sub(r'\1\2\3\4\5\6\7\n\n', t)))

    print('   - title abbreviations at line breaks...')
    df = df.assign(
        text=df.text.apply(lambda t: end_of_line_abbr.sub(r'\1\2\5\6 \3\4', t)))

    text_diff = ~df.text.isin(orig_df.text)
    if any(text_diff):
        changedf = df.loc[text_diff, :]
        print(f'{len(changedf)} of {len(df)} texts modified')

    # raw column will no longer be saved to finalized dataframe output
    # dataframes in `raw/` will have only `raw`
    # dataframes in `tmp/` will have both `raw` and `text`
    # dataframes in the parent dir, `_DATAFRAMES_DIRNAME/` will have only `text`
    df.pop('raw')
    df = df.assign(text=df.text.astype('string'))
    return df


def get_elapsed_time(start, end):
    elapsed_time = timedelta(seconds=round(
        end.timestamp()) - round(start.timestamp()))
    return elapsed_time


def pull_exclusions(df: pd.DataFrame,
                    excl_save_path: Path,
                    recheck: bool = False):

    print('  pulling excluded formats...')
    excl_df = pd.DataFrame(
        columns=['text_id', 'slice_id', 'excl_type', 'text',
                 'pile_set_name', 'pile_set_code'])
    loaded_from_file = False
    found_exclusions = False
    prev_excl_count = 0
    if excl_save_path.is_file() and not recheck:
        loaded_from_file = True
        prev_excl = pd.read_pickle(excl_save_path)
        excl_df = pd.concat([excl_df, prev_excl],
                            ignore_index=True)
        df = df.loc[~df.text_id.isin(excl_df.text_id)]
        prev_excl_count = len(excl_df)
        print(f'  -> {prev_excl_count} previously '
              'identified exclusions loaded from file')
    else:
        print('[No previous exclusion assessment found.]')

    # uninterpretable/unknown characters (could not be decoded)
    print('  looking for uninterpretable characters...')
    t0 = time.perf_counter()
    cannot_interpret = df.text.str.contains(_UNK_CHAR_STR)
    if any(cannot_interpret):
        found_exclusions = True
        unkchardf = df.loc[cannot_interpret, :].assign(excl_type='?unk')
        print(f'   +{len(unkchardf)} exclusions')
        excl_df = pd.concat([excl_df, unkchardf])
        df = df.loc[~cannot_interpret, :]
    t1 = time.perf_counter()
    print(f'= ?unk char excl ~~ {round(t1-t0, 2)} seconds')

    # wikitext
    t0 = time.perf_counter()
    df, wiki_df = exclude_wikitexts(df)
    if not wiki_df.empty:
        found_exclusions = True
        wiki_df = wiki_df.assign(excl_type='wiki')
        excl_df = pd.concat([excl_df, wiki_df])
    t1 = time.perf_counter()
    print(f'= wiki excl ~~ {round(t1-t0, 2)} seconds')

    # html source code
    t0 = time.perf_counter()
    df, html_df = exclude_html(df)
    if not html_df.empty:
        found_exclusions = True
        html_df = html_df.assign(excl_type='html')
        excl_df = pd.concat([excl_df, html_df])
    t1 = time.perf_counter()
    print(f'= html excl ~~ {round(t1-t0, 2)} seconds')

    # flag texts that contain technical seeming strings and exclude for now
    print('  looking for other messy text...')

    df, excl_df, found_exclusions = exclude_regex(
        df, excl_df, found_exclusions)

    #! currently unreachable (no calls of function as recheck)
    if recheck:
        if not found_exclusions:
            print('all previously flagged exclusions have been fixed by simple cleanup')
        elif len(excl_df) < len(df):
            print('some excluded texts have been fixed by simple cleanup')
        else:
            print('all prev exclusions remain')

    else:

        # * remove `raw` column for exclusion dataframes
        # (any added later due to failed parsing attempts will not
        #  have the `raw` column, and this info can always be
        # retrieved from dataframes in `raw/` if needed)
        excl_df = pop_unwanted_cols(excl_df)

        # only save if (new) texts were marked as exclusions
        #   or if no pre-existing file (to prevent researching later)
        if found_exclusions:
            print('saving...')
            excl_df.to_pickle(excl_save_path)
            print(f'  = {len(excl_df)} exclusions '
                  f'({len(excl_df) - prev_excl_count} new) saved to '
                  f'{get_print_path(excl_save_path)}')

        elif loaded_from_file:
            print('  = No additional exclusions found.')

        else:
            print('saving...')
            excl_df.to_pickle(excl_save_path)
            print('  = No exclusions found.')
        # if len(excl_df) > 0:
        #     print(f'e.g.:\n', excl_df.sample(1).text.iloc[0][:800])

    return df, excl_df


def exclude_regex(df, excl_df, found_excl):

    pattern_type_dict = {
        'json': json_regex,
        'code': code_regex,
        '_wrd': underscore_regex,
        'a0wrd': mixed_letter_digit_regex,
        'punc': midword_punc_regex
    }

    for excl_type_str, excl_regex in pattern_type_dict.items():
        t0 = time.perf_counter()
        excl_boolean = df.text.apply(lambda t: bool(excl_regex.search(t)))
        if any(excl_boolean):
            found_excl = True
            new_excl = df.loc[excl_boolean, :]
            new_excl = new_excl.assign(excl_type=excl_type_str)
            print(f'   +{len(new_excl)} {excl_type_str} exclusions')
            excl_df = pd.concat([excl_df, new_excl])
            df = df.loc[~df.text_id.isin(excl_df.text_id), :]
        t1 = time.perf_counter()
        print(f'= {excl_type_str} filtering ~~ {round(t1-t0, 2)} seconds')

    return df, excl_df, found_excl


def exclude_wikitexts(df):
    print('  looking for wikitext/wikimedia formatting...')
    wikidf = pd.DataFrame()

    is_wiki = df.text.apply(lambda t: bool(defwiki.search(t)))
    if any(is_wiki):
        wikidf = pd.concat([wikidf, df.loc[is_wiki, :]])
        df = df.loc[~is_wiki, :]

    maybe_wiki = (df.text.apply(lambda t: bool(wikipat.search(t))))
    if any(maybe_wiki):
        wikidf = pd.concat([wikidf, df.loc[maybe_wiki, :]])
        df = df.loc[~maybe_wiki, :]

    print(f'   +{len(wikidf)} exclusions')
    return df, wikidf


def exclude_html(df):
    print('  looking for any html...')
    html_df = pd.DataFrame()
    is_html = df.text.apply(
        lambda t: bool(likely_html.search(t)))

    if any(is_html):

        html_df = df.loc[is_html, :]
        df = df.loc[~is_html, :]

    print(f'   +{len(html_df)} exclusions')
    return df, html_df


# slicing up data for output
def slice_df(full_df):

    for subcorpus_code, df in full_df.groupby('pile_set_code'):
        subcorpus_name = df.pile_set_name.iat[0]
        data_orig_fpath = df.data_origin_fpath.iat[0]
        data_grp_str = data_orig_fpath.stem
        print(f'\n{len(df)} remaining {subcorpus_name} texts in '
              f'{data_grp_str} dataset\n'
              f'Slicing dataframe into smaller subsets of around {_SLDF_ROW_LIMIT} rows each'
              )
        # TODO : do this in a more elegant way...  pandas chunking?
        remaining_df = df.sort_values('text_id')
        slices = []
        # e.g. if limit were 1000:
        # slice off 1000 rows at a time until total is 2300 or less
        while len(remaining_df) > int(2.3*_SLDF_ROW_LIMIT):

            dfslice = remaining_df.iloc[:_SLDF_ROW_LIMIT, :]
            remaining_df = remaining_df.iloc[_SLDF_ROW_LIMIT:, :]
            slices.append(dfslice)
        # if 2400 split remaining: 2 slices of 1200
        # if 1202, split remaining: 2 slices of 610
        # if remaining df is 1200 rows or less:
        #   keep as is (no more slicing)
        if len(remaining_df) > int(1.15*_SLDF_ROW_LIMIT):

            half_remaining = int(len(remaining_df)/2)

            dfslice_penult = remaining_df.iloc[:half_remaining, :]
            slices.append(dfslice_penult)

            remaining_df = remaining_df.iloc[half_remaining:, :]

        # this must be outdented to catch smaller dataframes
        slices.append(remaining_df)
        slices_total_str = str(len(slices))
        zfill_len = len(slices_total_str)
        # Andrea Hummel on Feb 3, 2022 at 4:45 PM
        # * Note that this first save of the dataframe slices is *after* `pull_exclusions()`
        #   is called, so to get the full set of texts covered in the full dataframe, need
        #   to look at the union of the slices _and_ the corresponding exclusions dataframe.
        slice_labels = [str(x+1).zfill(zfill_len) for x in range(len(slices))]
        slice_info = pd.DataFrame(
            columns=['total_texts', 'first_text_id', 'last_text_id',
                     'tmp_slice_path', 'final_slice_path', 'conllu_path'],
            index=slice_labels)
        for i, zipped in enumerate(zip(slice_labels, slices)):
            #! slice numbering starts at 1, not 0 (cannot use i)
            slice_zfilled, sldf = zipped

            # reset index for each slice dataframe to range index
            #! must have `drop = True` bc otherwise inserts existing index as a column
            #! (and you can only do this 2x without renaming those columns before an error occurs)
            # * (original index from full df can be retrieved from `orig_text_id`)
            sldf.reset_index(drop=True, inplace=True)

            # update text ids
            sldf = create_ids(sldf, zfilled_slice_num=slice_zfilled)
            sldf = sldf.assign(slice_numstr=slice_zfilled)
            #! must reassign modified dataframe to the list of slices to save update
            slices[i] = sldf

            # save slice dataframe as compressed pickle
            outpath = get_dfpkl_outpath(
                data_grp_str, subcorpus_code,
                slice_id=slice_zfilled, is_tmp=True)
            sldf.to_pickle(outpath)

            # add info on slice to slice_info meta dataframe
            first_id = sldf.text_id.iloc[0]
            last_id = sldf.text_id.iloc[-1]
            texts_in_slice = len(sldf)
            conllu_path = get_conllu_outpath(
                data_grp_str, slice_zfilled, subcorpus_code)
            slice_info.loc[slice_zfilled, :] = [
                texts_in_slice, first_id, last_id,
                outpath.relative_to(_DESTINATION),
                outpath.parent.parent.joinpath(
                    outpath.name).relative_to(_DESTINATION),
                conllu_path.relative_to(_DESTINATION)]

        # //final_df_path = get_dfpkl_outpath(data_source_label, subcorpus_name)
        try:
            final_df_path = df.dataframe_fpath.iloc[0].relative_to(
                _DESTINATION)
        except ValueError:
            final_df_path = df.dataframe_fpath.iloc[0]
        slice_info = slice_info.assign(
            origin_filepath=data_orig_fpath,
            data_origin_group=data_grp_str,
            final_df_path=final_df_path,
            exclusions_path=get_dfpkl_outpath(
                data_grp_str, subcorpus_name, is_excl=True
            ).relative_to(_DESTINATION))
        slice_info.index.name = 'slice_number'
        # and save slice_info same directory when finished looping
        slice_info.to_csv(outpath.with_name(
            get_metadf_fname(data_grp_str)))
        print(slice_info)

        #! this needs to be its own loop so that all the slices can be saved
        #!   before any of them are processed
        #!   (which takes a long time and has a high likelihood of crashing)
        for sldf in slices:
            process_slice(sldf, slice_info)


def process_slice(sldf: pd.DataFrame, metadf: pd.DataFrame):
    slice_t0 = datetime.now()
    sldf = pop_unwanted_cols(sldf)
    slices_total_str = '?' if metadf.empty else len(metadf)

    # first_id_in_slice = sldf.text_id.iloc[0]
    # data_group, first_textid_in_slice = first_id_in_slice.split('_')[2:4]
    this_sl_numstr = sldf.slice_numstr[0]
    slice_int = int(this_sl_numstr)

    metadf.index = pd.to_numeric(metadf.index, downcast='unsigned')
    #! row index must be in [] to return dataframe instead of series
    this_sl_metadf = metadf.loc[[slice_int], :]

    subcorpus_code = sldf.pile_set_code[0]
    data_group = this_sl_metadf.data_origin_group.squeeze()
    try:
        data_group_cap = data_group.capitalize()
    # if loaded directly from sliced dataframes, may not be string
    except AttributeError:
        data_group_cap = str(data_group).zfill(2)
    slice_name = f'{subcorpus_code.capitalize()}{data_group_cap}_{this_sl_numstr}'
    print('=============================\n'
          f'Slice "{slice_name}" started\n  @ {slice_t0.ctime()}\n')

    this_sl_metadf = this_sl_metadf.assign(slice_name=slice_name,
                                           slice_number=this_sl_metadf.index,
                                           started_at=slice_t0.ctime())
    # // slice_info_row = slice_info_row.set_index('slice_id')
    this_sl_series = this_sl_metadf.squeeze()

    # parse slice and write to conllu output file
    conllu_path_record = Path(this_sl_series.conllu_path)

    # likely, conllu path will already be relative to _DESTINATION
    # if in 'puddin' dir somewhere, and can find the parent dir, should work as is

    local_conllu_path = (conllu_path_record
                         if _DESTINATION in conllu_path_record.parents
                         else _DESTINATION.joinpath(conllu_path_record))

    # if path isn't of format: .../puddin/[data group][slice num].conll/[conllu file]
    if local_conllu_path.parent.parent.name != 'puddin':
        local_conllu_path = get_conllu_outpath(
            data_group, sldf.slice_numstr[0], subcorpus_code)

    elif not local_conllu_path.parent.is_dir():
        local_conllu_path.parent.mkdir(parents=True)

    successful_df = stanza_parse(
        sldf, local_conllu_path, slice_int, slices_total_str)

    final_slice_path = Path(this_sl_series.final_slice_path)
    if _DESTINATION not in final_slice_path.parents:
        final_slice_path = _DESTINATION.joinpath(final_slice_path)

    successful_df.to_pickle(final_slice_path)
    slice_t1 = datetime.now()
    print(f'Finished writing parses to {this_sl_series.conllu_path}\n'
          f'  @ {slice_t1.ctime()}')
    delta = get_elapsed_time(slice_t0, slice_t1)
    print(f'    {delta} -- Slice parsing time')

    # // runtime = timedelta(seconds=round(
    # // slice.timestamp() - global_start_time.timestamp()))
    print(f'    {get_elapsed_time(_SCRIPT_START_TIME, datetime.now())} '
          '-- Current script runtime')

    # save version of dataframe for all texts actually processed
    this_sl_metadf = this_sl_metadf.assign(
        finished_at=slice_t1.ctime(),
        parsing_time=delta)

    this_sl_metadf = this_sl_metadf.set_index('slice_name')

    master_metadf_path = _DESTINATION.joinpath(
        f'{_DATAFRAMES_DIRNAME}/master_all-slices_index.csv')
    if master_metadf_path.is_file():
        master_slice_metadf = pd.read_csv(master_metadf_path)

    else:
        master_slice_metadf = pd.DataFrame()

    if 'slice_name' in master_slice_metadf.columns:
        master_slice_metadf = master_slice_metadf.set_index('slice_name')

    master_slice_metadf = pd.concat(
        [master_slice_metadf, this_sl_metadf])
    master_slice_metadf.to_csv(master_metadf_path)
    print('Info for fully processed slice added to master completed slice meta index:',
          get_print_path(master_metadf_path))

    if len(successful_df) == len(sldf):
        print('No skipped texts added to exclusions')
        return

    # if any texts failed,
    #   load previous exclusions file to add them
    # below note was originally in `slice_df()`  but still relevant here
    #     Feb 11, 2022 at 9:13 PM
    #       Currently, `excl_df` is not passed into this method. Instead of
    #     getting it via a redundant call to `pull_exclusions()`
    #     (that happens in `clean_df()` now) it is loaded from where it
    #     was saved either in the previous call, or in a previous run
    #     of the script.
    #     If for some reason the exclusions file cannot be found, run again.
    #        (but save with different name so as to not overwrite original.)
    excl_save_path = get_dfpkl_outpath(
        Path(this_sl_series.exclusions_path).stem, is_excl=True)
    if excl_save_path.is_file():
        excl_df = pd.read_pickle(excl_save_path)
        print('Adding skipped texts to', get_print_path(excl_save_path))

    else:
        backup_path = excl_save_path.with_name(excl_save_path.name
                                               .split('.', 1)[0]+'-alt.pkl.gz')
        if backup_path.is_file():
            excl_df = pd.read_pickle(backup_path)
        else:
            excl_df = pd.DataFrame()

        print('Exclusions file could not be found. Saving skipped texts to ',
              get_print_path(backup_path))

    # add any skipped texts/rows
    skip_df = sldf.loc[~sldf.text_id.isin(successful_df.text_id), :]
    if 'orig_text_id' in skip_df.columns:
        skip_df = skip_df.assign(excl_type='fail',
                                 slice_id=skip_df.text_id,
                                 text_id=skip_df.orig_text_id)
        skip_df.pop('orig_text_id')
    else:
        skip_df = skip_df.assign(excl_type='fail',
                                 slice_id=skip_df.text_id)

    excl_df = pd.concat([excl_df, skip_df], axis=0, ignore_index=True)
    print(f'{len(skip_df)} texts added to exclusions (with excl_type="fail"):')
    print(skip_df[['text_id', 'slice_id', 'excl_type']])

    # save exclusions df
    excl_df.to_pickle(excl_save_path)

### parsing functions ###


def stanza_parse(df: pd.DataFrame,
                 conllu_save_path: Path,
                 filenum: int = 0,
                 total_num_slices: str = 0):

    # TODO : change POS to XPOS; remove extra features?

    # really just a way to initiate a boolean series of the right length
    # all rows should be False at this point
    row_skipped = df.text.isna()

    num_texts_in_slice = len(df)
    print(f'Starting slice {filenum} of {total_num_slices}: '
          f'{num_texts_in_slice} texts in current slice')

    # open output file for conll formatted data
    print(f'  parsed data will be written to {conllu_save_path}')
    with conllu_save_path.open(mode='w') as conllout:

        # for each text in the pile subset slice...
        for position_in_slice, row_index in enumerate(df.index):
            # `position_in_slice` should only be used for ordinal/counting
            parse_t0 = datetime.now()
            row_df = df.loc[[row_index], :]

            text_id = row_df.text_id.squeeze()
            print(f'  {position_in_slice+1} of {num_texts_in_slice} '
                  f'in slice {filenum} (of {total_num_slices}): {text_id}')

            textstr = row_df.text.squeeze()

            # the text can be parsed with json.loads(), it's in json format,
            # which which will break stanza (likely) and is unwanted material (definitely)
            try:
                #! needs to be stripped bc quoted text can be interpreted as a json object
                __ = json.loads(textstr.strip('"\''))
            except json.decoder.JSONDecodeError:
                pass
            except ValueError:
                pass
            else:
                print(f'    in json format. Skipping.')
                row_skipped.loc[row_index] = True
                continue

            # create doc (with parsing)
            try:
                doc = _NLP(textstr)
            except RuntimeError:
                print('WARNING! Excluding unparsable text. (runtime error, '
                      'reason unknown). Skipping.')
                row_skipped.loc[row_index] = True

            else:
                doc = process_sentences(text_id, doc)

                # write conll formatted string of doc to output file
                conllout.write(_DOC2CONLL_TEXT(doc))

            parse_t1 = datetime.now()
            print('       time:', get_elapsed_time(parse_t0, parse_t1))

    # // t = datetime.now()
    # // print(f'Finished writing parses to {output_path}\n  @ {t.ctime()}')

    # // delta = timedelta(seconds=round(
    # //     t.timestamp() - global_start_time.timestamp()))
    # // print(f'  current script runtime: {delta}')

    successful_df = df.loc[~row_skipped, :]
    print(f'= {len(successful_df)} of '
          f'{num_texts_in_slice} texts successfully parsed.')

    return successful_df


def process_sentences(text_id: str,
                      doc: stanza.models.common.doc.Document):
    """function to add metacomments to a stanza document, 
    plus check the parses for obvious/easy to fix errors,
    and then returns that dock 

    Args:
        text_id (str): 
            identifier string for given document
        doc (stanza.models.common.doc.Document): 
            stanza document of parses for given text

    Returns:
        stanza.models.common.doc.Document: updated staza doc
    """
    print('    - processing sentences...')
    # // text_id = row_df.text_id.squeeze()
    if len(doc.sentences) == 0:
        print(f'!! No sentences in document {text_id}')
        print(f'    text: {doc.text}')
    # check for line breaks in sentence text string
    doc = confirm_parse(doc)
    sent_zfill = len(str(len(doc.sentences)))
    sent_id = ''
    # add comments to sentences (info pulled from dataframe)
    for enumi, sentence in enumerate(doc.sentences):
        enumi += 1
        #! ignore s.id--these will be off if sentences with line breaks were broken up above
        if enumi == 1:
            # "newdoc id" will be the text_id from the pile subset
            sentence.add_comment(f'# newdoc id = {text_id}')

        # "sent_id" will be doc/text id with _[sentence number] appended
        sent_id = f'{text_id}_{str(enumi).zfill(sent_zfill)}'
        sentence.add_comment(f'# sent_id = {sent_id}')
        if enumi == 1:
            print('     ', sent_id, '\n       ...')

        # remove line breaks and duplicated white space characters with single space
        text = remove_breaks(sentence.text)
        # this adds the full text string to the output file
        sentence.add_comment(f'# text = {text}')
    if sent_id:
        print('     ', sent_id)
    return doc


def confirm_parse(doc):

    for s in doc.sentences:
        text = s.text.strip()
        # if sentence has line breaks...
        if len(text.split('\n')) > 1:
            # print('\n===> Line breaks found: attempting remediation...')
            ix = doc.sentences.index(s)
            doc.sentences = try_redoc(ix, doc.sentences)
            # print('------------')
    return doc


def remove_breaks(textstr):
    """takes in a sentence string and returns the string with
        new lines and duplicated whitespace characters replaced by ' '. """

    cleantext = solonew_or_dupwhite.sub(r' ', textstr.strip())
    cleantext = extra_newlines.sub('\n\n', cleantext)
    return cleantext


def try_redoc(ix, sent_list):
    """takes in a single sentence string, either reformatted or
        the original text but as a single unit, and attempts to parse it.
        If the model generates a different parse than the existing,
        that will be returned. Otherwise the original is kept.

        Args:
            ix (int): index of original sentence object in existing sentences list
            sent_list (list): existing list of sentence objects

        Returns:
            sent_list (list): list of sentence objects;
            input list with new sentences inserted if any parse changes,
            otherwise with cleaned text replacing original sentence text.
        """

    text = sent_list[ix].text.strip()

    ''' NOTE: previously, the regex was stronger and caught 
        any case of sentence end punc before \n
        OR any case of capitals/sentence "starts" after \n.
        However, this splits up proper nouns that happen 
        to land at a \n into different sentences,
        so it's weaker now and requires *both*.
        This is motivated by the fact that better edited sources
        will have both, and it is better to preserve clean edited materials
        than trash them in a likely futile attempt to salvage messy data
        (and messy in wildly varying ways, too) e.g.:
        not split:
        [no punctuation]
        If ImmoNatie would be an object, it would be..An Iphone
        If ImmoNatie would be a car, it would be..A Mini
        ---
        No more jokes about French
        No more jokes about women
        CHICKENS ONLY
        [no approved "start" char]
        ...
        3 star - getting closer but need revisions or a different take on your design
        4 star - design is being considered
        5 star - made the final cut!
        [to preserve]:
        ...shock loss for second seed Tommy
        Robredo.
        ---
        ... the first one will be Duke Nukem
        Trilogy.
        ---
        The latest Batman movie... drove overall
        Hollywood box office weekend sales of the top 12 titles... .
        ---
        Thanks,
        Meyrav Levine.
        ---
        ...opposed the idea as the start of reform process promoted by
        King Abdullah that they fear will liberalize the stringent system.
        split:
        A motion was made by Joseph Marsh and seconded by Jason Forbes to adjourn at 4:44 p.m.
        Jamison called for a voice vote on that motion and all members voted yes.
        :\n is treated as a sentence break, whereas midline : is not; e.g.
        Sentences:
        (0) EDIT:
        (1) Freedreno is up, performance is on par with existing...
        Sentences:
        (0) NOTE: Apply a light coat of Premium Long Life Grease XG-1-C ...'''
    if linebreak_is_sent.search(text):
        # print('multiple sentences found:')
        print(f'    original sentence {ix} split into:')
        plausible_sep_text = linebreak_is_sent.sub(r'\1\3\n\n\2\4', text)
        plausible_sep_text = remove_breaks(plausible_sep_text)
        new_sentences = _NLP(plausible_sep_text).sentences
        for s in new_sentences:
            print(f'    + {s.text}')
    else:
        # // print('    Line breaks removed from sentence text:')
        new_sentences = _NLP(remove_breaks(text)).sentences
        # // print(f'      {new_sentences[0].text}')

    return sent_list[:ix] + new_sentences + sent_list[ix+1:]


def parse_arg_inputs():

    parser = argparse.ArgumentParser(
        description='script to convert scrambled pile data from raw jsonlines format '
        'into dependency parsed conllu files. Note that if neither input files nor a '
        'search glob expression is specified, the calling directory AND subdirectories '
        'will be searched for jsonlines files. '
        '*Required packages: stanza (stanford), unidecode, zlib, and pandas')

    parser.add_argument(
        '-i', '--input_file', default=[],
        type=Path, action='append', dest='input_files',
        help=('path(s) for input file(s). Can be `.jsonl` or `.pkl(.gz).` '
              'If not specified, script will seek all applicable files '
              'the scope of the calling directory or directory specified with -s flag.'))

    parser.add_argument(
        '-g', '--glob_expr',
        type=str, default='*/*/data/pile/**/*jsonl',
        help=("glob expression for file input selection. Ignored if `-i` flag is used."
              "*Must be enclosed in '/\" in order to be input as string "
              "instead of multiple files!* "
              "WARNING: In order to access other directories, "
              "the search is now relative to the anchor (i.e. /). "
              "Therefore beginning the expression with '**' is highly discouraged"
              "--it will take a very long time to look through EVERYthing."
              "Defaults to '*/*/data/pile/**/*jsonl', which will seek a 'data/' directory "
              "2 levels down from the anchor of the file system with subdir 'pile/'"
              "and all subsequent paths ending in 'jsonl'"))

    parser.add_argument(
        '-d', '--destination',
        type=Path, default=Path.cwd(),
        help=('Path to destination for output directory, `puddin/`, '
              'which will contain all output files, at all stages of processing: '
              '`_DATAFRAMES_DIRNAME/`, `_EXCLUSIONS_DIRNAME/`, `....conll/`. '
              'Results in, e.g., `[DESTINATION]/puddin/_DATAFRAMES_DIRNAME/`'))

    parser.add_argument(
        '-c', '--corpus_selection',
        default='Pile-CC',
        type=str,
        help=('option to specify the pile set. Default pile set: "Pile-CC".'))

    parser.add_argument(
        '-R', '--Reprocess',
        default=False, action='store_true',
        help=('option to skip step looking for existing progress entirely and reprocess files, '
              'even if more processed outputs have already been created. Intended for debugging.'))

    parser.add_argument(
        '-S', '--reSlice',
        default=False, action='store_true',
        help=('option to seek furthest stage of processing for *full* dataframes, '
              'but ignore any previously created slices. '
              'Note the --Reprocess option nullifies this one. '
              '\n**Will wipe the selected data\'s subdirectory in `_DATAFRAMES_DIRNAME/slices/`. '
              '(same effect as just manually deleting the slices before running the script.)'))

    parser.add_argument(
        '-o', '--output_size',
        default=9999, type=int,
        help=('option to specify the target output size (in number of texts) '
              'for the sliced dataframes and the eventual corresponding conllu files. '
              'Defaults to 9999 texts, which yields ~99 slices per original pile data '
              'file for Pile-CC subcorpus. '
              'Note that this will only apply to *new* (or redone) slicing.')
    )

    args = parser.parse_args()

    #! the input string has to be quoted for it to remain a string
    #!   instead of a list of files (unrecognized arguments)
    ge = args.glob_expr.strip('\'"')
    if not args.input_files and ge.startswith('/'):
        ge = ge[1:]
        print('WARNING: invalid glob expression (Nonrelative paths not supported).\n'
              ' > Leading "/" removed. Will search for:',
              ge)
    args.glob_expr = ge

    return args


if __name__ == '__main__':
    _main()
    end_time = datetime.now()
    elapsed = str(timedelta(seconds=round(
        end_time.timestamp() - _SCRIPT_START_TIME.timestamp())))
    print(f'\n*******************************\nScript Completed: {end_time.ctime()}\n '
          f'= total elapsed time: {elapsed}')
