import re

likely_html = re.compile(r'<(\w*).*>[^<]*</\1>')

defwiki = re.compile(r'<nowiki>')
wikipat = re.compile(r'[{[]{2,}[^|}\]]+\|[^}\]]*\}{2,}')
# wt0 = re.compile(r"\*")
# wt1 = re.compile(r"(['(\s[])\w+/([\w{}]+)")
# wt2 = re.compile(r"\[{2}spoiler:(.+?)\]{2}")
# wt3 = re.compile(r"(\[{2})(note\]{2})(.+?)\1/\2")
# wt4 = re.compile(r"\[{2}/?(\w+|\w+:[\w &]+)\]{2}")
# wt5 = re.compile(r'\[{2}\S+\s([^\]]+)\]{2}')
# wt6 = re.compile(r"\[=([^=]+)=\]")
# wt7 = re.compile(r"([\{'])\1(\w+)\|([A-Z][\w]+)(\})\4")
# wt8 = re.compile(r"\||(?<=@)/")
# wt9 = re.compile(r"([{'])\1\w+/(\w+)([}'])\3")
# wt10 = re.compile(r"\{{2}([^{}]+)\}{2}")
# wt11 = re.compile(r"('{2,})([^']+'?[^']+?)\1")
# wt12 = re.compile(r'\n{4,}')

bracket_url = re.compile(r'\[url=[^\]]*]([^[]*)\[/url\]')
likely_url = re.compile(
    r'https?://\S*\s|www\.\S*\s|[\w\d]+\.[\w\d]+\.[\w\d]+\S*\s|http://www\.\w+\.\w{2:3}'
    # r'(\(?\[?(?:https?)://w{0,3}\S*[^\s./]{2,}\.[^\s./]{2,}[\./\@]\S*[/:\@]\S*)'
)

# exclusion regex patterns
# letters interspersed with numbers in the same "word"
mixed_letter_digit_regex = re.compile(
    r'\d*[a-z]+\d+[a-z]*\d*[a-z]*'
    r'|\d{3:}[a-z]+[a-z]*\d*[a-z]*', re.IGNORECASE)

# (exclusion filters)
# single "word" containing `_` or other non acceptible mid-word punch
underscore_regex = re.compile(r'[\w]*?_[\w]+?')
midword_punc_regex = re.compile(
    r'\b[a-z]+[^\w\s\-\'/\\&@]+?[a-zA-Z]+\b')

# for cleaning --> .sub(r'\1\3 \2\4', str)
missing_space_regex = re.compile(r'(?# lowercaseUppercase with no \s)([a-z]+)([A-Z])'
                                 r'|(?# word-edge punc with no \s)([a-z][.!?,;:]+)([A-Z])')
# a single instance of code declaration
code_regex = re.compile(
    r'(=|[=!><][=!><])\s?(self|true|false|\w+\.?\w*)',
    re.IGNORECASE)

# a single instance of json like dictionary formatting
json_regex = re.compile(r'{"\w+":{"\w+":')

# cleaning regex patterns
# abbreviations that can be followed by r"\. [A-Z]" without signaling end of sentence
# only `Aa` capitalization is considered sentence start, not `AA` (another abbr.)
end_of_line_abbr = re.compile(
    r'(?:(Mr|M[sx]|Messrs|Mmes|[SG]en|[FS]t|Re[vp]|Pr(?:es|of)|Supe?|Capt'
    r'|Asst|Ms?gr|Engr?|Assoc|Arb|Assemb|Pharm?|Hon|i\.e|e\.g|ca?'
    r'|(?<![A-Z])[A-Z](?![A-Z]))(e?s?\.[^\w\n]?)\n([^\n\w]?[A-Z]))'
    r'|(?<!\n)\n([^\n\w]?[A-Z]{2,})'
    r'|(Jan|Feb|Mar|Apr|Ju[nl]|Aug|Sept?|Nov|Oct|Dec)(\.?)\n(?=\d)')
punc_only = re.compile(
    r'(?# full line nonword chars only )^([\W_]+)$'
    r'|(?# any punc/non`\n`ws repeated 4+)(_|[^\w\n])(\2{4,})'
    r'|(?# punc/non`\n`ws except . repeated 4)([^a-z\d.\n])(\4{3})'
    r'|(?# punc/non`\n`ws except .!?$*= or blank repeated 3)([^a-z\d.!?$=* \n])(\6{2})',
    re.MULTILINE | re.IGNORECASE)
linebreak_is_sent = re.compile(
    r'(?:(?#1--> )([^A-Z\n]{3,}[.?!;][\'"?! \t\f\v\r]*|\.{4,})\n[ \t\f\v\r]*(?#2--> )([(#["\']?[A-Z]|\W*?\d+\W*?\w))'
    r'|(?:(?#3--> )(\D[.;:][\'"?! \t\f\v\r]*)\n[ \t\f\v\r]*(?#4--> )([\(\#\["\']?[A-Z]|[\#\[\(]\d+[\)\]]))')

solonew_or_dupwhite = re.compile(r'(?<!\n)(\n)(?!\n)|([ \t\f\v\r])\2+')
extra_newlines = re.compile(r'\n{3,}')

# nonbreaking_colon = re.compile(r'\d+?:\n\d+?')
# linebreak_is_sent = re.compile(
#     r'([\w\d][.?!][\'"?! ]*?)\n+|([^,;:)\]\)/-])\n+'
#     + r'([A-Z][^A-Z]|[(#["\'][A-Z]|\W*?\d+.*?\w)')

start_chars = re.compile(
    r'^[A-Z][^A-Z\n]|^[\(\["\'][A-Z]|^[\W]*$')
sent_end_punc = re.compile(r'([.?!][\'"?!]?)')
