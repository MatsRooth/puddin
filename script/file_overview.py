# coding=utf-8


from pathlib import Path
import time

def _size_round(size: int):

    if size >= 10**8:
        unit = 'G'
        power = 9
    elif size >= 10**5:
        unit = 'M'
        power = 6
    elif size >= 10**2:
        unit = 'K'
        power = 3
    else:
        unit = ''
        power = 0

    return f'{round(size / (10**power), 1):.1f} {unit}B'


def _main():
    val_by_grp_path = Path('/share/compling/data/puddin/info/validation_by_group/')
    print(f'Files in {val_by_grp_path}',time.ctime(), sep='\n @ ', end='\n* * * * * * * * * * * * * *\n')
    for f in val_by_grp_path.joinpath('status-overview').glob('*pcc_status*gz'): 
        print(str(Path(*f.parts[-3:])))
        print('modified:',time.ctime(f.stat().st_mtime))
        print('size:',_size_round(f.stat().st_size))
        name = f.name
        grp = name.capitalize()[:2]
        name = f'Pcc{grp}_status-info.pkl.gz'
        print(f'renamed: {Path(*f.with_name(name).parts[-3:])}')
        f.rename(f.with_name(name))

        print('-----------------------------')
    
        
if __name__ == '__main__': 
    _main()