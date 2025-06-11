# fmt: off

'''
Conversion of text from a Crystallographic Information File (CIF) format to
unicode. CIF text is neither unicode nor bibtex/latex code.

Rules for character formatting in CIF files are specified at:
https://www.iucr.org/resources/cif/spec/version1.1/semantics
'''

import html
import re

subs_dict = {
    '\r': '',            # Windows line ending
    '\t': ' ',           # tabs

    r'\a': '\u03b1',    # alpha
    r'\b': '\u03b2',    # beta
    r'\g': '\u03b3',    # gamma
    r'\d': '\u03b4',    # delta
    r'\e': '\u03b5',    # epsilon
    r'\z': '\u03b6',    # zeta
    r'\h': '\u03b7',    # eta
    r'\q': '\u03b8',    # theta
    r'\i': '\u03b9',    # iota
    r'\k': '\u03ba',    # kappa
    r'\l': '\u03bb',    # lambda
    r'\m': '\u03bc',    # mu
    r'\n': '\u03bd',    # nu
    r'\x': '\u03be',    # xi
    r'\o': '\u03bf',    # omicron
    r'\p': '\u03c0',    # pi
    r'\r': '\u03c1',    # rho
    r'\s': '\u03c3',    # sigma
    r'\t': '\u03c4',    # tau
    r'\u': '\u03c5',    # upsilon
    r'\f': '\u03c6',    # phi
    r'\c': '\u03c7',    # chi
    r'\y': '\u03c8',    # psi
    r'\w': '\u03c9',    # omega
    r'\A': '\u0391',    # Alpha
    r'\B': '\u0392',    # Beta
    r'\G': '\u0393',    # Gamma
    r'\D': '\u0394',    # Delta
    r'\E': '\u0395',    # Epsilon
    r'\Z': '\u0396',    # Zeta
    r'\H': '\u0397',    # Eta
    r'\Q': '\u0398',    # Theta
    r'\I': '\u0399',    # Ioto
    r'\K': '\u039a',    # Kappa
    r'\L': '\u039b',    # Lambda
    r'\M': '\u039c',    # Mu
    r'\N': '\u039d',    # Nu
    r'\X': '\u039e',    # Xi
    r'\O': '\u039f',    # Omicron
    r'\P': '\u03a0',    # Pi
    r'\R': '\u03a1',    # Rho
    r'\S': '\u03a3',    # Sigma
    r'\T': '\u03a4',    # Tau
    r'\U': '\u03a5',    # Upsilon
    r'\F': '\u03a6',    # Phi
    r'\C': '\u03a7',    # Chi
    r'\Y': '\u03a8',    # Psi
    r'\W': '\u03a9',    # Omega

    r'\%a': '\u00e5',   # a-ring
    r'\/o': '\u00f8',   # o-slash
    r'\?i': '\u0131',   # dotless i
    r'\/l': '\u0142',   # Polish l
    r'\&s': '\u00df',   # German eszett
    r'\/d': '\u0111',   # barred d

    r'\%A': '\u00c5',   # A-ring
    r'\/O': '\u00d8',   # O-slash
    r'\?I': 'I',         # dotless I
    r'\/L': '\u0141',   # Polish L
    r'\&S': '\u1e9e',   # German Eszett
    r'\/D': '\u0110',   # barred D

    r'\%': '\u00b0',           # degree
    r'--': '\u2013',           # dash
    r'---': '\u2014',          # single bond
    r'\\db': '\u003d',         # double bond
    r'\\tb': '\u2261',         # triple bond
    r'\\ddb': '\u2248',        # delocalized double bond
    r'\\sim': '~',
    r'\\simeq': '\u2243',
    r'\\infty': '\u221e',      # infinity

    r'\\times': '\u00d7',
    r'+-': '\u00b1',           # plusminus
    r'-+': '\u2213',           # minusplus
    r'\\square': '\u25a0',
    r'\\neq': '\u2660',
    r'\\rangle': '\u3009',
    r'\\langle': '\u3008',
    r'\\rightarrow': '\u2192',
    r'\\leftarrow': '\u2190',

    r"\'A": '\u00c1',  # A acute
    r"\'E": '\u00c9',  # E acute
    r"\'I": '\u00cd',  # I acute
    r"\'O": '\u00d3',  # O acute
    r"\'U": '\u00da',  # U acute
    r"\'Y": '\u00dd',  # Y acute
    r"\'a": '\u00e1',  # a acute
    r"\'e": '\u00e9',  # e acute
    r"\'i": '\u00ed',  # i acute
    r"\'o": '\u00f3',  # o acute
    r"\'u": '\u00fa',  # u acute
    r"\'y": '\u00fd',  # y acute
    r"\'C": '\u0106',  # C acute
    r"\'c": '\u0107',  # c acute
    r"\'L": '\u0139',  # L acute
    r"\'l": '\u013a',  # l acute
    r"\'N": '\u0143',  # N acute
    r"\'n": '\u0144',  # n acute
    r"\'R": '\u0154',  # R acute
    r"\'r": '\u0155',  # r acute
    r"\'S": '\u015a',  # S acute
    r"\'s": '\u015b',  # s acute
    r"\'Z": '\u0179',  # Z acute
    r"\'z": '\u017a',  # z acute
    r"\'G": '\u01f4',  # G acute
    r"\'g": '\u01f5',  # g acute
    r"\'K": '\u1e30',  # K acute
    r"\'k": '\u1e31',  # k acute
    r"\'M": '\u1e3e',  # M acute
    r"\'m": '\u1e3f',  # m acute
    r"\'P": '\u1e54',  # P acute
    r"\'p": '\u1e55',  # p acute
    r"\'W": '\u1e82',  # W acute
    r"\'w": '\u1e83',  # w acute
    r'\;A': '\u0104',  # A ogonek
    r'\;a': '\u0105',  # a ogonek
    r'\;E': '\u0118',  # E ogonek
    r'\;e': '\u0119',  # e ogonek
    r'\;I': '\u012e',  # I ogonek
    r'\;i': '\u012f',  # i ogonek
    r'\;U': '\u0172',  # U ogonek
    r'\;u': '\u0173',  # u ogonek
    r'\;O': '\u01ea',  # O ogonek
    r'\;o': '\u01eb',  # o ogonek
    r'\.C': '\u010a',  # C dot above
    r'\.c': '\u010b',  # c dot above
    r'\.E': '\u0116',  # E dot above
    r'\.e': '\u0117',  # e dot above
    r'\.G': '\u0120',  # G dot above
    r'\.g': '\u0121',  # g dot above
    r'\.I': '\u0130',  # I dot above
    r'\.Z': '\u017b',  # Z dot above
    r'\.z': '\u017c',  # z dot above
    r'\.A': '\u0226',  # A dot above
    r'\.a': '\u0227',  # a dot above
    r'\.O': '\u022e',  # O dot above
    r'\.o': '\u022f',  # o dot above
    r'\.B': '\u1e02',  # B dot above
    r'\.b': '\u1e03',  # b dot above
    r'\.D': '\u1e0a',  # D dot above
    r'\.d': '\u1e0b',  # d dot above
    r'\.F': '\u1e1e',  # F dot above
    r'\.f': '\u1e1f',  # f dot above
    r'\.H': '\u1e22',  # H dot above
    r'\.h': '\u1e23',  # h dot above
    r'\.M': '\u1e40',  # M dot above
    r'\.m': '\u1e41',  # m dot above
    r'\.N': '\u1e44',  # N dot above
    r'\.n': '\u1e45',  # n dot above
    r'\.P': '\u1e56',  # P dot above
    r'\.p': '\u1e57',  # p dot above
    r'\.R': '\u1e58',  # R dot above
    r'\.r': '\u1e59',  # r dot above
    r'\.S': '\u1e60',  # S dot above
    r'\.s': '\u1e61',  # s dot above
    r'\.T': '\u1e6a',  # T dot above
    r'\.t': '\u1e6b',  # t dot above
    r'\.W': '\u1e86',  # W dot above
    r'\.w': '\u1e87',  # w dot above
    r'\.X': '\u1e8a',  # X dot above
    r'\.x': '\u1e8b',  # x dot above
    r'\.Y': '\u1e8e',  # Y dot above
    r'\.y': '\u1e8f',  # y dot above
    r'\(A': '\u0102',  # A breve
    r'\(a': '\u0103',  # a breve
    r'\(E': '\u0114',  # E breve
    r'\(e': '\u0115',  # e breve
    r'\(G': '\u011e',  # G breve
    r'\(g': '\u011f',  # g breve
    r'\(I': '\u012c',  # I breve
    r'\(i': '\u012d',  # i breve
    r'\(O': '\u014e',  # O breve
    r'\(o': '\u014f',  # o breve
    r'\(U': '\u016c',  # U breve
    r'\(u': '\u016d',  # u breve
    r'\=A': '\u0100',  # A macron
    r'\=a': '\u0101',  # a macron
    r'\=E': '\u0112',  # E macron
    r'\=e': '\u0113',  # e macron
    r'\=I': '\u012a',  # I macron
    r'\=i': '\u012b',  # i macron
    r'\=O': '\u014c',  # O macron
    r'\=o': '\u014d',  # o macron
    r'\=U': '\u016a',  # U macron
    r'\=u': '\u016b',  # u macron
    r'\=Y': '\u0232',  # Y macron
    r'\=y': '\u0233',  # y macron
    r'\=G': '\u1e20',  # G macron
    r'\=g': '\u1e21',  # g macron
    r'\^A': '\u00c2',  # A circumflex
    r'\^E': '\u00ca',  # E circumflex
    r'\^I': '\u00ce',  # I circumflex
    r'\^O': '\u00d4',  # O circumflex
    r'\^U': '\u00db',  # U circumflex
    r'\^a': '\u00e2',  # a circumflex
    r'\^e': '\u00ea',  # e circumflex
    r'\^i': '\u00ee',  # i circumflex
    r'\^o': '\u00f4',  # o circumflex
    r'\^u': '\u00fb',  # u circumflex
    r'\^C': '\u0108',  # C circumflex
    r'\^c': '\u0109',  # c circumflex
    r'\^G': '\u011c',  # G circumflex
    r'\^g': '\u011d',  # g circumflex
    r'\^H': '\u0124',  # H circumflex
    r'\^h': '\u0125',  # h circumflex
    r'\^J': '\u0134',  # J circumflex
    r'\^j': '\u0135',  # j circumflex
    r'\^S': '\u015c',  # S circumflex
    r'\^s': '\u015d',  # s circumflex
    r'\^W': '\u0174',  # W circumflex
    r'\^w': '\u0175',  # w circumflex
    r'\^Y': '\u0176',  # Y circumflex
    r'\^y': '\u0177',  # y circumflex
    r'\^Z': '\u1e90',  # Z circumflex
    r'\^z': '\u1e91',  # z circumflex
    r'\"A': '\u00c4',  # A diaeresis
    r'\"E': '\u00cb',  # E diaeresis
    r'\"I': '\u00cf',  # I diaeresis
    r'\"O': '\u00d6',  # O diaeresis
    r'\"U': '\u00dc',  # U diaeresis
    r'\"a': '\u00e4',  # a diaeresis
    r'\"e': '\u00eb',  # e diaeresis
    r'\"i': '\u00ef',  # i diaeresis
    r'\"o': '\u00f6',  # o diaeresis
    r'\"u': '\u00fc',  # u diaeresis
    r'\"y': '\u00ff',  # y diaeresis
    r'\"Y': '\u0178',  # Y diaeresis
    r'\"H': '\u1e26',  # H diaeresis
    r'\"h': '\u1e27',  # h diaeresis
    r'\"W': '\u1e84',  # W diaeresis
    r'\"w': '\u1e85',  # w diaeresis
    r'\"X': '\u1e8c',  # X diaeresis
    r'\"x': '\u1e8d',  # x diaeresis
    r'\"t': '\u1e97',  # t diaeresis
    r'\~A': '\u00c3',  # A tilde
    r'\~N': '\u00d1',  # N tilde
    r'\~O': '\u00d5',  # O tilde
    r'\~a': '\u00e3',  # a tilde
    r'\~n': '\u00f1',  # n tilde
    r'\~o': '\u00f5',  # o tilde
    r'\~I': '\u0128',  # I tilde
    r'\~i': '\u0129',  # i tilde
    r'\~U': '\u0168',  # U tilde
    r'\~u': '\u0169',  # u tilde
    r'\~V': '\u1e7c',  # V tilde
    r'\~v': '\u1e7d',  # v tilde
    r'\~E': '\u1ebc',  # E tilde
    r'\~e': '\u1ebd',  # e tilde
    r'\~Y': '\u1ef8',  # Y tilde
    r'\~y': '\u1ef9',  # y tilde
    r'\<C': '\u010c',  # C caron
    r'\<c': '\u010d',  # c caron
    r'\<D': '\u010e',  # D caron
    r'\<d': '\u010f',  # d caron
    r'\<E': '\u011a',  # E caron
    r'\<e': '\u011b',  # e caron
    r'\<L': '\u013d',  # L caron
    r'\<l': '\u013e',  # l caron
    r'\<N': '\u0147',  # N caron
    r'\<n': '\u0148',  # n caron
    r'\<R': '\u0158',  # R caron
    r'\<r': '\u0159',  # r caron
    r'\<S': '\u0160',  # S caron
    r'\<s': '\u0161',  # s caron
    r'\<T': '\u0164',  # T caron
    r'\<t': '\u0165',  # t caron
    r'\<Z': '\u017d',  # Z caron
    r'\<z': '\u017e',  # z caron
    r'\<A': '\u01cd',  # A caron
    r'\<a': '\u01ce',  # a caron
    r'\<I': '\u01cf',  # I caron
    r'\<i': '\u01d0',  # i caron
    r'\<O': '\u01d1',  # O caron
    r'\<o': '\u01d2',  # o caron
    r'\<U': '\u01d3',  # U caron
    r'\<u': '\u01d4',  # u caron
    r'\<G': '\u01e6',  # G caron
    r'\<g': '\u01e7',  # g caron
    r'\<K': '\u01e8',  # K caron
    r'\<k': '\u01e9',  # k caron
    r'\<j': '\u01f0',  # j caron
    r'\<H': '\u021e',  # H caron
    r'\<h': '\u021f',  # h caron
    r'\>O': '\u0150',  # O double acute
    r'\>o': '\u0151',  # o double acute
    r'\>U': '\u0170',  # U double acute
    r'\>u': '\u0171',  # u double acute
    r'\,C': '\u00c7',  # C cedilla
    r'\,c': '\u00e7',  # c cedilla
    r'\,G': '\u0122',  # G cedilla
    r'\,g': '\u0123',  # g cedilla
    r'\,K': '\u0136',  # K cedilla
    r'\,k': '\u0137',  # k cedilla
    r'\,L': '\u013b',  # L cedilla
    r'\,l': '\u013c',  # l cedilla
    r'\,N': '\u0145',  # N cedilla
    r'\,n': '\u0146',  # n cedilla
    r'\,R': '\u0156',  # R cedilla
    r'\,r': '\u0157',  # r cedilla
    r'\,S': '\u015e',  # S cedilla
    r'\,s': '\u015f',  # s cedilla
    r'\,T': '\u0162',  # T cedilla
    r'\,t': '\u0163',  # t cedilla
    r'\,E': '\u0228',  # E cedilla
    r'\,e': '\u0229',  # e cedilla
    r'\,D': '\u1e10',  # D cedilla
    r'\,d': '\u1e11',  # d cedilla
    r'\,H': '\u1e28',  # H cedilla
    r'\,h': '\u1e29',  # h cedilla
    r'\`A': '\u00c0',  # A grave
    r'\`E': '\u00c8',  # E grave
    r'\`I': '\u00cc',  # I grave
    r'\`O': '\u00d2',  # O grave
    r'\`U': '\u00d9',  # U grave
    r'\`a': '\u00e0',  # a grave
    r'\`e': '\u00e8',  # e grave
    r'\`i': '\u00ec',  # i grave
    r'\`o': '\u00f2',  # o grave
    r'\`u': '\u00f9',  # u grave
    r'\`N': '\u01f8',  # N grave
    r'\`n': '\u01f9',  # n grave
    r'\`W': '\u1e80',  # W grave
    r'\`w': '\u1e81',  # w grave
    r'\`Y': '\u1ef2',  # Y grave
    r'\`y': '\u1ef3',  # y grave
}

superscript_dict = {
    '0': '\u2070',  # superscript 0
    '1': '\u00b9',  # superscript 1
    '2': '\u00b2',  # superscript 2
    '3': '\u00b3',  # superscript 3
    '4': '\u2074',  # superscript 4
    '5': '\u2075',  # superscript 5
    '6': '\u2076',  # superscript 6
    '7': '\u2077',  # superscript 7
    '8': '\u2078',  # superscript 8
    '9': '\u2079',  # superscript 9
}

subscript_dict = {
    '0': '\u2080',  # subscript 0
    '1': '\u2081',  # subscript 1
    '2': '\u2082',  # subscript 2
    '3': '\u2083',  # subscript 3
    '4': '\u2084',  # subscript 4
    '5': '\u2085',  # subscript 5
    '6': '\u2086',  # subscript 6
    '7': '\u2087',  # subscript 7
    '8': '\u2088',  # subscript 8
    '9': '\u2089',  # subscript 9
}


def replace_subscript(s: str, subscript=True) -> str:

    target = '~'
    rdict = subscript_dict
    if not subscript:
        target = '^'
        rdict = superscript_dict

    replaced = []
    inside = False
    for char in s:
        if char == target:
            inside = not inside
        elif not inside:
            replaced += [char]
        # note: do not use char.isdigit - this also matches (sub/super)scripts
        elif char in rdict:
            replaced += [rdict[char]]
        else:
            replaced += [char]

    return ''.join(replaced)


def multiple_replace(text: str, adict) -> str:
    rx = re.compile('|'.join(map(re.escape, adict)))

    def one_xlat(match):
        return adict[match.group(0)]

    return rx.sub(one_xlat, text)


def format_unicode(s: str) -> str:
    """Converts a string in CIF text-format to unicode.  Any HTML tags
    contained in the string are removed.  HTML numeric character references
    are unescaped (i.e. converted to unicode).

    Parameters:

    s: string
        The CIF text string to convert

    Returns:

    u: string
        A unicode formatted string.
    """

    s = html.unescape(s)
    s = multiple_replace(s, subs_dict)
    tagclean = re.compile('<.*?>')
    return re.sub(tagclean, '', s)


def handle_subscripts(s: str) -> str:
    s = replace_subscript(s, subscript=True)
    s = replace_subscript(s, subscript=False)
    return s
