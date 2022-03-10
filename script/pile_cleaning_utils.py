#!/usr/bin/env python
# coding=utf-8
import wikitextparser as wtp
from pile_regex_imports import *

def clean_wikitexts(df):
    print('  looking for wikitext/wikimedia formatting...')
    wikidf = None

    is_wiki = df.text.apply(lambda t: bool(defwiki.search(t)))
    if any(is_wiki):
        wikidf = df.loc[is_wiki, :]
        print(
            f'  extracting text from {len(wikidf)} known wikitext formatting...')
        print('text_ids:', wikidf.text_id.to_list())
        wikidf = wikidf.assign(
            text=wikidf.text.apply(lambda t: wtp.parse(t).plain_text()).astype('string'))
        still_wiki = wikidf.text.apply(lambda t: bool(defwiki.search(t)))
        print(wikidf[['text', 'raw']])

        if any(still_wiki):
            wikidf.loc[still_wiki, 'text'] = (wikidf.loc[still_wiki, 'text']
                                              .apply(lambda t: _reformat_wiki(t)))

        df.loc[is_wiki, :] = wikidf

    maybe_wiki = (df.text.apply(lambda t: bool(wikipat.search(t))))
    if any(maybe_wiki):
        wikidf = df.loc[maybe_wiki, :]
        print(f'  cleaning {len(wikidf)} possibly wikitext formatted texts...')
        cleaned_text = wikidf.text.apply(lambda t: _reformat_wiki(t))
        wikidf = wikidf.assign(text=cleaned_text.astype('string'))
        df.loc[maybe_wiki, :] = wikidf

    if wikidf is None:
        print('    [none found]')
    return df


def _reformat_wiki(t):

    # add vertical space between bullet points
    t = wt0.sub('\n\n', t)
    t = wt1.sub(r'\1\2', t)
    t = wt2.sub(r'\1', t)
    t = wt3.sub(r' (\3)', t)
    t = wt4.sub('\n\n', t)
    # remove links, internal or url
    t = wt5.sub(r'\1', t)
    t = wt6.sub(r'\1', t)
    t = wt7.sub(r'\1\1\2 \3\4\4', t)
    t = wt8.sub('', t)
    t = wt9.sub(r'\2', t)
    t = wt10.sub(r'\1', t)
    t = wt11.sub(r'\2', t)
    t = wt12.sub('\n\n\n', t)

    return t