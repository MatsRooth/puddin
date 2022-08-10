# puddin
The Pile + universal dependencies & decreased internal noise

## About

This repository contains the code used to produce a *truly* cleaned and filtered subset (of your choosing) of [The Pile](https://pile.eleuther.ai/) parsed into [CoNLL-U Format](https://universaldependencies.org/format.html#conll-u-format)

## Set Up

### The environment

To create 
the `puddin` (ana)conda environment 
(in your home directory), 
from `puddin/` run: 
  
    $ conda env create -f puddin_env.yml

Or from any other location, run:

    $ conda env create -f [full/path/to/puddin_env.yml]

Using an anaconda environment is recommended, but if that is not possible, the python modules required can be found in `puddin/requirements.txt`, and this can be used with `pip`: 
    
    pip install -r requirements.txt

The corpus data to be parsed needs to be: 
1. accessible from the working directory, and 
2. in jsonlines format, i.e. `.jsonl` file extensions.

All outputs will be saved to subdirectories created in the directory where the main script is called.   

### The data

The Pile original data files can be obtained [here](https://mystic.the-eye.eu/public/AI/pile/).

While this was designed to be run on The Pile, it could easily be extended to run on other plain text language data in jsonlines format. (The main tweaks required would be to alter the path names/expectations/conventions, and the column names expected for the jsonlines file, and the dictionary indicating the abbreviated format of the subset name.)



## Example Usage

*Note that the python executable must point to python3*
```
../puddin$ python script/parse_pile.py
```
If the original jsonlines data files are within the scope of the working directory, this command will iterate over all located `.jsonl` files. 

To specify which data to parse, the options `--input_file`/`-i` or `--glob_expr`/`-g` can be used.

By default, the code will seek the furthest progress point for a given data set (i.e. with the same original `.jsonl` name): To override this completely, use `--Reprocess`/`-R`. To repartition the data only (no reprocessing steps), use `--reSlice`/`-S`.

    ../puddin$ python script/parse_pile.py -h
    usage: parse_pile.py [-h] [-i INPUT_FILES] [-g GLOB_EXPR]
                        [-c CORPUS_SELECTION] [-R] [-S] [-o OUTPUT_SIZE]

    script to convert scrambled pile data from raw jsonlines format into
    dependency parsed conllu files. Note that if neither input files nor a
    search glob expression is specified, the calling directory AND
    subdirectories will be searched for jsonlines files. *Required packages:
    stanza (stanford), unidecode, zlib, and pandas

    optional arguments:
    -h, --help            show this help message and exit
    -i INPUT_FILES, --input_file INPUT_FILES
                            path(s) for input file(s). Can be `.jsonl` or
                            `.pkl(.gz).` If not specified, script will seek
                            all applicable files the scope of the calling
                            directory or directory specified with -s flag.
    -g GLOB_EXPR, --glob_expr GLOB_EXPR
                            glob expression for file input selection. Is
                            superseded by the `-i` flag*Must be enclosed in
                            '/" in order to be input as string instead of
                            multiple files!* Should be relative to current
                            working directory and specify at least 1
                            directory that contains valid data files.
                            Defaults to '**/*jsonl' (files ending in jsonl in
                            current working directory or any of its
                            subdirectories, recursively. '*/*jsonl' would
                            look in the working dir and its immediate
                            subdirs.
    -c CORPUS_SELECTION, --corpus_selection CORPUS_SELECTION
                            option to specify the pile set. Default pile set:
                            "Pile-CC".
    -R, --Reprocess       option to skip step looking for existing progress
                            entirely and reprocess files, even if more
                            processed outputs have already been created.
                            Intended for debugging.
    -S, --reSlice         option to seek furthest stage of processing for
                            *full* dataframes, but ignore any previously
                            created slices. Note the --Reprocess option
                            nullifies this one. **Will wipe the selected
                            data's subdirectory in `pile_tables/slices/`.
                            (same effect as just manually deleting the slices
                            before running the script.)
    -o OUTPUT_SIZE, --output_size OUTPUT_SIZE
                            option to specify the target output size (in
                            number of texts) for the sliced dataframes and
                            the eventual corresponding conllu files. Defaults
                            to 9999 texts, which yields ~99 slices per
                            original pile data file for Pile-CC subcorpus.
                            Note that this will only apply to *new* (or
                            redone) slicing.