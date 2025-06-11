import re


def normalize_file_whitespace(text):
    """remove initial and final whitespace on each line, replace any interal
    whitespace with one space, and remove trailing blank lines"""

    lines_out = []
    for line in text.strip().splitlines():
        lines_out.append(re.sub(r'\s+', ' ', line.strip()))
    return '\n'.join(lines_out)


def filecmp_ignore_whitespace(f1, f2):
    """Compare two files ignoring all leading and trailing whitespace, amount of
    whitespace within lines, and any trailing whitespace-only lines."""

    with open(f1) as f1_o, open(f2) as f2_o:
        same = (normalize_file_whitespace(f1_o.read()) ==
                normalize_file_whitespace(f2_o.read()))
    return same
