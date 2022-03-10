from os import write
import re
from pathlib import Path
from sys import argv

import pyconll


def main():

    metasplit_re = re.compile(r'(\# text = )')
    startparse_re = re.compile(r'(\n1\t\S)')
    brk_re = re.compile(r'\n+')
    allowed_re = re.compile(r'^# sent|^# newdoc|^# text|^\d{,3}\t')
    fix_dir = Path(argv[1])
    out_dir = fix_dir.parent.joinpath(f'{fix_dir.stem}_fixed.conll')
    if not out_dir.is_dir():
        out_dir.mkdir()

    for fix_file in fix_dir.glob('*.conllu'):
        texts_fixed = 0
        file_text_list = []
        orig_text = fix_file.read_text()
        
        sents = re.split(r'(\n\n)', orig_text)
        for sent in sents:
            text_list = []
            # if delimiter or no problem lines, don't need to do all the splitting and rejoining
            if sent == '\n\n' or all(allowed_re.search(s) or s == '' for s in sent.split('\n')):
                text = sent
            
            else:
                # splits at `# text = `
                sent_chunks = metasplit_re.split(sent, maxsplit=1)

                for chunk in sent_chunks:
                    # for actual sentence info chunks:
                    if not chunk.startswith(('\n', '# sent_id',)):
                        # split at `\n1\t\S+`; parts[0] should be "# text" field
                        # parts[1:2] should be [parse start delimiter: 1    [word], [dep parse lines]]
                        parts = startparse_re.split(chunk, maxsplit=1)
                        # only want to remove excess line breaks from `# text` field
                        if brk_re.search(parts[0]):
                            parts[0] = brk_re.sub(' ', parts[0])
                            chunk = ''.join(parts)
                            texts_fixed += 1

                    # append chunk (meta or parse info) to sentence text list
                    text_list.append(chunk)

                # TODO : the text isn't getting broken up enough... tweak the regex
                #   for each sentence conll object, try to parse it with pyconll
                text = ''.join(text_list)
            try:
                __ = pyconll.load_from_string(text)
            # this is for temporary devel purposes
            # TODO :  if this can't be fixed, error handling will
            #   need to be added to grewSearchDir.py
            except:
                print('Conll Error for\n>>>>\n', text[:100], text[-100:])
            else:
                file_text_list.append(text)
        
        write_string = ''.join(file_text_list)
        # one final sanity check test: 
        #   see if any (nonblank) lines start with anything other than 0-9 or #
        if any(re.search(r'^[^\d#]', line) for line in write_string.split('\n')): 
            print('problem lines still!!!')
        print(texts_fixed, '`# text` fields fixed in', fix_file.name)
        with out_dir.joinpath(f'{fix_file.name}').open('w') as new:
            new.write(write_string)


if __name__ == '__main__':

    main()
