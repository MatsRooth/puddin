#!/usr/bin/env python3

"""
quicky script that compares two conda environments

can be handy for debugging differences between two environments

This could be made much cleaner and more flexible -- but it does the job.

Please let me know if you extend or improve it.

AUTHOR: Christopher H. Barker Chris.Barker@noaa.gov
LICENSE: Public domain -- do with it what you will
(But it would be nice to give me credit if it helps you)

UPDATE: Ben Taylor
I changed the print statements and had to add
a conversion from bytes to string for Python 3.

I just ran this in Python 3.7.3 and seemd to work well.

UPDATE: Andrea Hummel @ Aug 7, 2022
I applied my own tweaks simply because I could.
Functionality has largely not been modified.
"""

import argparse
import subprocess

import pandas as pd

_PRINT_SEPLINE_CHAR = '.'


def _main():
    env1_pkg_info = get_env_pkg_info(ENV_1)
    env2_pkg_info = get_env_pkg_info(ENV_2)

    in_one, in_two, diff_version, same_version = compare_envs(
        env1_pkg_info, env2_pkg_info)

    env1_df = _convert_to_df(env1_pkg_info)
    env2_df = _convert_to_df(env2_pkg_info)

    _print_header(f"Only in `{ENV_1}`:")
    _print_report(env1_df, in_one)

    _print_header(f"Only in `{ENV_2}`:")
    _print_report(env2_df, in_two)

    _print_header("In both, with different versions:")
    _print_diff_version_df(env1_df, env2_df, diff_version)

    _print_header("(In both, same version:)")
    _print_report(env1_df, same_version)


def _convert_to_df(pkg_info: dict):

    return pd.DataFrame(
        pkg_info, index=['version', 'build', 'channel']).transpose()


def _print_header(header):
    print("\n", header, "\n", _PRINT_SEPLINE_CHAR*len(header), sep='')


def get_env_pkg_info(env_name: str):
    """get version, build, and channel info for all packages contained in given conda env

    Args:
        env_name (str): name of conda environment

    Returns:
        dict: dictionary of tuples detailing package info.
                Item format: `package_name: (version, build, channel)`
    """
    cmd = "conda list -n " + env_name
    pkg_list = subprocess.check_output(cmd, shell=True)
    pkg_list = pkg_list.decode('utf-8')
    # process the package list
    pkg_info_dict = {}
    for line in pkg_list.split('\n'):
        line = line.strip()
        if not line or line[0] == '#':
            continue
        parts = line.split()
        pkg, version, build = parts[:3]
        if build == '<pip>':
            channel = "pip"
        else:
            channel = "defaults" if len(parts) < 4 else parts[3]
        pkg_info_dict[pkg] = (version, build, channel)

    return pkg_info_dict


def compare_envs(env1_pkg_dict: dict, env2_pkg_dict: dict):
    """compare packages across given conda environments:
    takes in 2 dictionaries and returns 4 comparison lists

    Args:
        env1_pkg_dict (dict): ENV_1 package info
        env2_pkg_dict (dict): ENV_2 package info

    Returns:
        list: packages in ENV_1 only,
        list: packages in ENV_2 only,
        list: packages in both with different versions,
        list: packages in both with same version
    """
    pkg_set_1 = set(env1_pkg_dict)
    pkg_set_2 = set(env2_pkg_dict)

    in_both = pkg_set_1 & pkg_set_2
    in_one = list((pkg_set_1 ^ pkg_set_2) & pkg_set_1)
    in_two = list((pkg_set_1 ^ pkg_set_2) & pkg_set_2)

    diff_version = []
    same_version = []
    for pkg in in_both:
        if env1_pkg_dict[pkg] != env2_pkg_dict[pkg]:
            diff_version.append(pkg)
        else:
            same_version.append(pkg)

    return in_one, in_two, diff_version, same_version


def _print_report(pkgs_df: pd.DataFrame, pkgs_to_print):

    info_to_print = pkgs_df.loc[pkgs_to_print, :].sort_index()
    print(info_to_print.to_string())



def _print_diff_version_df(env1_df: pd.DataFrame, env2_df: pd.DataFrame, pkgs: list):
    env1_df = env1_df.loc[pkgs, :].assign(env=ENV_1).reset_index()
    env1_df.columns = ['package', 'version', 'build', 'channel', 'env']
    env1_df = env1_df.set_index(['package', 'env'])

    env2_df = env2_df.loc[pkgs, :].assign(env=ENV_2).reset_index()
    env2_df.columns = ['package', 'version', 'build', 'channel', 'env']
    env2_df = env2_df.set_index(['package', 'env'])

    df_str = pd.concat([env1_df, env2_df]).sort_index().to_string()
    df_lines = df_str.splitlines()
    print(df_lines.pop(0))
    print(df_lines.pop(0))
    print_lines = []
    for line in df_lines:
        print_lines.append(line+'\n' if line.startswith(' ') else line)

    print('\n'.join(print_lines))


def _parse_args():
    parser = argparse.ArgumentParser(
        description=('Compare 2 conda environments. '
                     'Prints info on package contents and versions compared between 2 envs.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'env',
        type=str, default=None,
        help=('name of first conda environment. Required.')
    )

    parser.add_argument(
        '-c', '--compare_env',
        type=str, default='base',
        help=('name of second conda environment. '
              'Also required, but defaults to "base" environment.'))

    args = parser.parse_args()
    return args.env, args.compare_env


if __name__ == "__main__":

    ENV_1, ENV_2 = _parse_args()
    _main()
