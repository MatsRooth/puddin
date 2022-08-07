import argparse
from collections import namedtuple
import re
import sys
from pathlib import Path
from subprocess import run as sp_run

SELECT_RAW_REGEX = re.compile(
    r'(?<=_)eng_([valtes\d]{2,4}_)\d+\.\d+_x(?=\d+)',
    re.IGNORECASE)


def _main():

    args = _parse_args()
    if not args.conll_dir.is_dir():
        sys.exit(f'{args.conll_dir} is not an existing directory.')
    for i in conllu_id_iter(**args):
        print(i)


def _parse_args():

    parser = argparse.ArgumentParser(
        description=(''),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'conll_dir',
        type=Path, default=None,
        help=(
            'path to directory containing conllu files to be assessed.')
    )

    parser.add_argument(
        'id_unit',
        type=str,
        choices=['doc', 'document', 'sent', 'sentence'],
        default='document',
        help=('string indicating what id to pull.'
              'Options are: doc(ument), sent(ence)')
    )

    parser.add_argument(
        '-r', '--raw', action='store_true', default=False,
        help=('Option to reconstruct "raw" text ids from conllu document ids '
              '(i.e. as they appear in the initial dataframes before slicing and parsing)'
              'for compatibility purposes.'))

    return parser.parse_args()


def conllu_id_iter(conll_dir: Path, id_unit: str,
                   #    reconstruct_raw=False,
                   iterate=True):
    # ^ use this to get descriptive statistics for sentences and docs
    grep_str = None
    if id_unit.startswith('doc'):
        grep_str = '# newdoc id ='

    elif id_unit.startswith('sent'):
        grep_str = '# sent_id ='

    elif id_unit.startswith('pat'):
        grep_str = '# pattern_match ='

    if not grep_str:
        sys.exit('No valid id unit specified. No ids pulled.')

    # TODO : make this parallel as well?
    for conllu_file in conll_dir.glob('*.conllu'):
        id_iter = None
        # unit_count = int(sp_run(['egrep', '-c', grep_str, conllu_file],
        #                         capture_output=True,
        #                         universal_newlines=True,
        #                         check=True).stdout.strip())

        if iterate:
            id_iter = _generate_id_match(grep_str, conllu_file)

        yield conllu_file.stem, id_iter


def _generate_id_match(grep_str, conllu_file):

    id_pattern = re.compile(r'(?<= = )(.*)\n')

    egrep_output = sp_run(['egrep', grep_str, conllu_file],
                          capture_output=True,
                          universal_newlines=True,
                          check=True).stdout

    id_match_iter = id_pattern.finditer(egrep_output)

    # if reconstruct_raw:
    #     for text in reconstruct_raw_iter(id_match_iter):
    #         yield text.raw

    # else:
    for id_match in id_match_iter:
        yield id_match.group().strip()


def reconstruct_raw_iter(parsed_doc_ids):
    '''generator function to yield ?? tuples of both parsed and reconstructed "raw" text/document id'''
    id_translate = namedtuple('TextID', ['conll_id', 'raw_id'])
    for doc_id in parsed_doc_ids:
        if doc_id:
            yield id_translate(doc_id, SELECT_RAW_REGEX.sub(r'\1', doc_id))
        else:
            yield id_translate(doc_id, '')


if __name__ == '__main__':
    _main()
