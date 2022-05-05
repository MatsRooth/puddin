import re
from pathlib import Path
DATA_DIR = Path('/home/arh234/data/puddin/')
SLICES_DIR = DATA_DIR.joinpath('pile_tables/slices')
EXCL_DIR = DATA_DIR.joinpath('pile_exclusions')

for conll_dir in DATA_DIR.glob('*.conll'):

    empties = tuple(c for c in conll_dir.glob(
        '*.conllu') if c.stat().st_size == 0)

    data_grp_name = conll_dir.stem
    final_sldf_dir = SLICES_DIR.joinpath(data_grp_name)

    if not empties:
        print(f'No empty conllu files in {conll_dir}')

    else:
        for empty_conllu in empties:

            # find "final" slice dfs and remove them as well
            empty_id = re.sub(r'(?<=-)0*(\d)', r'*\1',
                              empty_conllu.stem.split('_')[-1])
            print(f'empty file found: {empty_conllu.relative_to(DATA_DIR)}')
            error_df_paths = tuple(
                final_sldf_dir.glob(f'*{empty_id}*'))
            if error_df_paths:
                error_df_path = error_df_paths[0]
                print(
                    f'corresponding slice dataframe: {error_df_path.relative_to(DATA_DIR)}\n')
                error_df_path.unlink()

            empty_conllu.unlink()
