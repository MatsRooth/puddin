# coding=utf-8
# !/home/arh234/.conda/envs/puddin/bin/python
"""
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
"""
import argparse

# import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

# from multiprocessing_logging import install_mp_handler
import pandas as pd

from validate_data_group import (
    VALID_EXCL_DIR_NAME,
    assess_data_group,
    dur_round,
    _format,
)

# assign monikers for logging actions
# _debug = logging.debug
#! do not use logging.inform! It just causes problems with the output. Just use print().
# _warn = logging.warning
# _err = logging.error
# _crash = logging.critical


def _parse_args():

    parser = argparse.ArgumentParser(
        description=("program to validate puddin build"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        default="/share/compling/data/puddin",
        help=(
            "top level directory of puddin build. Should contain subdirectories for:"
            "raw, final, and sliced dataframes (default: pile_tables/), dataframes for "
            "all excluded texts for each data group (default: pile_exclusions/), "
            "parsed conllu files sorted by data source group (i.e. for each input "
            ".jsonlines file; default: Pcc[GRP].conll/), and an info/ dir "
            "containing the cleaned processing meta data."
        ),
    )

    parser.add_argument(
        "-g",
        "--data_group",
        type=str,
        action="append",
        dest="data_grps",
        default=[],
        help=(
            "option to restrict program to only run on given data groups. "
            "String should match stem of original data files. "
            "Can be used as many times as desired."
        ),
    )

    # parser.add_argument(
    #     '-l', '--log_level',
    #     choices=['debug', 'info', 'warning', 'error'],
    #     type=str, default='warning',
    #     help=('option to specify the verbosity of logging output'))

    # log_level_defs = {'debug': logging.DEBUG,
    #                   'info': logging.INFO,
    #                   'warning': logging.WARNING,
    #                   'error': logging.ERROR}

    args = parser.parse_args()

    return (
        args.data_dir,
        args.data_grps,
    )  # log_level_defs[args.log_level]


def _main():
    print(
        f'\n{time.strftime("%Y-%m-%d  %I:%M%p")}: Starting Puddin Validation of {_DATA_GRPS if _DATA_GRPS else "ALL data groups"}...'
    )
    # > Load meta info dataframe
    meta_info_path, meta = _load_meta_info()
    # For each row (i.e. slice) compare the text ids found in the files at
    # the following paths: raw, final, conllu. Make sure any missing
    # from conllu are in exclusions dataframe.
    all_groups_list = _assess_files(meta)

    if not all_groups_list:
        print(
            "<!> ERROR! Data validation failed. Could not find files "
            f"using paths specified in {meta_info_path}."
        )
        sys.exit(1)
    all_df = pd.concat(all_groups_list).sort_values("conll_id")

    _save_missing_info(all_df)

    print(
        "\n ==========================================\n|",
        time.strftime("%Y-%m-%d  %I:%M%p\n|"),
        _format(f"Assessment Complete:\n{all_df.loc[:,~all_df.columns.str.endswith('text')].sample(5)}"),
    )
    df_lines = (
        all_df.value_counts(["success", "data_group"])
        .sort_index()
        .to_string()
        .splitlines()
    )
    print_lines = ["\n" + df_lines.pop(0) + "\n"]
    for line in df_lines:
        print_lines.append(line + "\n" if line.startswith(" ") else line)

    print(
        " __________________________\n| Successful Parse Totals:"
        + _format("\n * * * * * *" + "\n".join(print_lines))
    )
    combined_results_path = _INFO_DIR.joinpath("all-pcc-texts_status.pkl.gz")
    if combined_results_path.is_file() and combined_results_path.stat().st_size > 0:
        prior_mtime_str = time.strftime(
            "%Y-%m-%d_%I%M%p", time.localtime(combined_results_path.stat().st_mtime)
        )
        new_name = combined_results_path.with_name(
            f"prior_pcc-validation-status_{prior_mtime_str}.pkl.gz"
        )
        prev_output = combined_results_path.rename(new_name)
    print(
        _format(
            f">> Total Successfully Parsed Texts: {all_df.success.value_counts()[True]}"
            f"\nPrior output renamed: {prev_output}"
            f"\nSaving compiled results to {combined_results_path} ..."
        )
    )
    all_df.to_pickle(combined_results_path)
    print("Finished.", time.strftime("%Y-%m-%d  %I:%M%p"), sep="\n")


def _load_meta_info():
    meta_info_path = _INFO_DIR.joinpath("completed-puddin_meta-index.pkl")
    print(
        f'{time.strftime("%Y-%m-%d  %I:%M%p")}: Loading processing meta info from {meta_info_path} ...'
    )
    if not meta_info_path.is_file():
        print(f"<!> ERROR! {meta_info_path} does not exist. Data cannot be assessed.")
        sys.exit(1)
    meta = pd.read_pickle(meta_info_path)

    if _DATA_GRPS:
        meta = meta.loc[(meta.data_origin_group.isin(_DATA_GRPS)), :]

        if meta.empty:
            print(f"<!> ERROR! No processing data found for sources: {_DATA_GRPS}.")
            sys.exit(1)

    was_overwritten = meta.slice_name.duplicated(keep="last")
    if any(was_overwritten):
        owdf = (
            meta.loc[
                :,
                meta.columns.isin(
                    ["slice_name", "finished_at", "conllu_mtime", "conllu_path"]
                ),
            ]
            .assign(overwritten=was_overwritten)
            .sort_values("conllu_path")
        )
        slices_with_dups = owdf.slice_name[owdf.overwritten].unique()
        list_str = "  ~ " + " and ".join(", ".join(slices_with_dups).rsplit(" ", 1))
        print(
            f"ERROR! Multiple records found for {len(slices_with_dups)} "
            f'slices:\n{list_str}\n{"."*len(list_str)}\n{owdf}'
        )
    # apparently some `slice_name` values are not zfilled? so can't sort by slice_name
    return meta_info_path, meta


def _assess_files(meta):
    all_groups_list = []
    bad_excl_list = []
    excl_dir = DATA_DIR.joinpath(meta.exclusions_path.iloc[-1]).parent
    # > not including `parents=True` bc it really should be a problem if
    # >   the indicated exclusions directory (the parent) cannot be found
    excl_dir.joinpath(VALID_EXCL_DIR_NAME).mkdir(exist_ok=True)
    tables_dir = DATA_DIR.joinpath(meta.final_df_path.iloc[-1]).parent

    grp_args_iter = (
        (grp, df, tables_dir, DATA_DIR)
        for grp, df in meta.groupby("data_origin_group")
        if not df.empty
    )

    input_count = len(meta.data_origin_group.unique())
    # > set pool `processes` argument to number of _available_ cpus
    # > OR number of files to be searched, whichever is smaller
    cpus = min(len(os.sched_getaffinity(0)), input_count)
    print(
        "\n _______________________\n|",
        time.strftime("%Y-%m-%d  %I:%M%p\n|"),
        _format(f"processing {input_count} inputs with {cpus} CPUs..."),
    )
    _start = time.perf_counter()

    multiprocessing.set_forkserver_preload(
        [
            "pull_ids_from_conll",
            "logging",
            "pd",
            "sys",
            "os",
            "DATA_DIR",
            "_inform",
            "_warn",
            "_err",
        ]
    )

    with multiprocessing.Pool(processes=cpus) as pool:

        # NOTE: if process function takes more than 1 argument:
        #       use a. snippet "expand_argument_inputs" and `imap(_unordered)`
        #       or  b. `starmap` -> does not require printing step after, but slower
        results = pool.imap_unordered(_star_assess_in_parallel, grp_args_iter)

        zfill_len = len(str(input_count))
        width = 10  # = data group
        #! this is required to actually get the processes to run
        run_count = 0
        for result in results:
            # > if assessment returned None;
            # >     i.e. could not open raw dataframe file (or other failure?)
            # > do not increase `run_count` for "non(e)-result"
            if result is None:
                continue
            group_info, excl_info, group, dur = result
            run_count += 1
            # > add dataframe for current group (gdf) to list of group dataframes
            all_groups_list.append(group_info)
            bad_excl_list.append(excl_info)

            if run_count == 1:
                print(
                    "\n _______________________\n|", time.strftime("%Y-%m-%d  %I:%M%p")
                )
                col_head_line = "\t" + "-" * width
                print(
                    (
                        "  task  |  time  \tdata group\tsuccessful\texclusions\n"
                        f" ------ | ------ {col_head_line*3}"
                    ).expandtabs(3)
                )
                
            was_success = group_info.success
            successful = str(group_info.value_counts(["success"])[True]) if any(was_success) else 0
            exclusions = str(group_info.value_counts(["success"])[False]) if any(~was_success) else 0
            print(
                (
                    f"{str(run_count).zfill(zfill_len).center(8)}|"
                    f"{dur.rjust(7)} \t"
                    f"{group.ljust(width)}\t"
                    f"{successful.ljust(width)}\t"
                    f"{exclusions.ljust(width)}"
                ).expandtabs(3)
            )

        total_inputs_processed = run_count
        # ? Is there a better way to do this? ^^ Like, some "run" or "start" or "join" method?

    _end = time.perf_counter()
    total_time = _end - _start
    print(
        "\n _______________________\n|",
        time.strftime("%Y-%m-%d  %I:%M%p\n|"),
        _format(
            f"= {total_inputs_processed} data groups validated in "
            f"{dur_round(total_time)}"
        ),
        "\n ````````````````````````````````````",
    )
    _save_bad_excl_info(bad_excl_list, excl_dir)

    return all_groups_list


def _star_assess_in_parallel(args):
    return _assess_in_parallel(*args)


def _assess_in_parallel(grp: str, info: pd.DataFrame, tables_dir: Path, data_dir: Path):

    _grp_start = time.perf_counter()

    rawdf_path = _check_meta_info(grp, info, tables_dir, data_dir)
    if not rawdf_path:
        return None

    grp_info, inv_excl_df = assess_data_group(grp, info, rawdf_path, data_dir)
    _grp_finish = time.perf_counter()
    time_str = dur_round(_grp_finish - _grp_start)
    return grp_info, inv_excl_df, grp, time_str


def _check_meta_info(grp, info, tables_dir, data_dir):
    """sanity check paths in meta processing info _for each data group_"""
    # if more than 1 path shown for finalized unsliced dataframe for given data source
    if len(info.final_df_path.unique()) != 1:
        print(
            "WARNING! More than 1 path found for unsliced dataframe for %s data source",
            grp,
        )
    if len(info.origin_filepath.unique()) != 1:
        print("WARNING! different paths showing for source file")

    findf_path = data_dir.joinpath(info.final_df_path.iloc[0])
    rawdf_path = Path(findf_path.parent, "raw", findf_path.name)
    if not rawdf_path.is_file():
        print("ERROR! Raw/initial dataframe not found. " f"Invalid path: {rawdf_path}")
        return None

    if findf_path.parent != tables_dir:
        print(
            _format(
                f"{grp} dataframe paths not in {tables_dir}. \n"
                f"   Is this file supposed to be included?: {findf_path}"
            )
        )

    return rawdf_path


def _save_bad_excl_info(bad_excl_list, excl_path):
    if any(not df.empty for df in bad_excl_list):
        bad_excl = pd.concat(bad_excl_list)
        print(
            " _______________________\n|",
            _format(f"WARNING! Erroneous Exclusions Found!\n{bad_excl.index}"),
        )
        bad_excl.to_json(
            _INFO_DIR.joinpath("excl_mistakes.json"), orient="index", indent=4
        )
    else:
        print(
            " _______________________\n|",
            f'No "false" exclusions found in {excl_path.relative_to(DATA_DIR.parent)}/',
        )


def _save_missing_info(all_df):
    if any(all_df.missing):
        all_missing_path = _INFO_DIR.joinpath("missing_texts.csv")
        log_message = _format(
            (
                "WARNING! Some original texts were unaccounted for!\n"
                "Missing text info for all data groups saved as "
                f"{all_missing_path.relative_to(DATA_DIR)}/.json"
            )
        )
        print(" _______________________\n|", log_message)
        missing_df = all_df.loc[all_df.missing, :]
        missing_df.to_csv(all_missing_path)
        missing_df.reset_index().set_index(
            ["data_group", "raw_id"], append=True
        ).to_json(all_missing_path.with_suffix(".json"), orient="index", indent=4)
    else:
        print(" _______________________\n| " + "All original texts accounted for ^_^")


if __name__ == "__main__":

    START = time.perf_counter()

    DATA_DIR, _DATA_GRPS = _parse_args()
    _INFO_DIR = DATA_DIR.joinpath("info")

    multiprocessing.set_start_method("forkserver")

    # logging.basicConfig(
    #     level=_log_level,
    #     # format='[%(levelname)s] %(asctime)s\n  > %(message)s',
    #     format=(' _______________________\n| %(levelname)s: %(asctime)s\n'
    #            '|    %(pathname)s:%(funcName)s:%(lineno)d\n| . . . . . . .\n   %(message)s'),
    #     datefmt="%Y-%m-%d %I:%M%p",
    #     # stream=sys.stdout,
    #     filename=str(Path('/share/compling/projects/puddin', 'logs',
    #                       f'validation{time.strftime("%Y-%m-%d_%H:%M")}.log')) )
    # install_mp_handler()
    _MP_LOGGER = multiprocessing.log_to_stderr()
    # debug = 10, info = 20, warning = 30, error = 40, critical = 50(?)
    _MP_LOGGER.setLevel(30)

    _main()

    END = time.perf_counter()
    print(f">> Total assessment time: {dur_round(END - START)} seconds")
