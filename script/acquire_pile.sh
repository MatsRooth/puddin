#!/bin/bash

# start from data/pile/train/ dir:

# wget https://mystic.the-eye.eu/public/AI/pile/train/00.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/01.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/02.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/03.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/04.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/05.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/06.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/07.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/08.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/09.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/10.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/11.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/12.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/13.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/14.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/15.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/16.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/17.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/18.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/19.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/20.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/21.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/22.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/23.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/24.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/25.jsonl.zst
wget https://mystic.the-eye.eu/public/AI/pile/train/26.jsonl.zst
wget https://mystic.the-eye.eu/public/AI/pile/train/27.jsonl.zst
wget https://mystic.the-eye.eu/public/AI/pile/train/28.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/train/29.jsonl.zst

unzstd 00.jsonl.zst
unzstd 01.jsonl.zst
unzstd 02.jsonl.zst
unzstd 03.jsonl.zst
unzstd 04.jsonl.zst
unzstd 05.jsonl.zst
unzstd 06.jsonl.zst
unzstd 07.jsonl.zst
unzstd 08.jsonl.zst
unzstd 09.jsonl.zst
unzstd 10.jsonl.zst
unzstd 11.jsonl.zst
unzstd 12.jsonl.zst
unzstd 13.jsonl.zst
unzstd 14.jsonl.zst
unzstd 15.jsonl.zst
unzstd 16.jsonl.zst
unzstd 17.jsonl.zst
unzstd 18.jsonl.zst
unzstd 19.jsonl.zst
unzstd 20.jsonl.zst
unzstd 21.jsonl.zst
unzstd 22.jsonl.zst
unzstd 23.jsonl.zst
unzstd 24.jsonl.zst
unzstd 25.jsonl.zst
unzstd 26.jsonl.zst
unzstd 27.jsonl.zst
unzstd 28.jsonl.zst
unzstd 29.jsonl.zst

# cd ..
# wget https://mystic.the-eye.eu/public/AI/pile/test.jsonl.zst
# wget https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst

# unzstd test.jsonl.zst
# unzstd val.jsonl.zst

echo "finished at $date"