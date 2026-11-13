"""rmitem.py FILE SIG [SIG...]: delete each Rust item whose first line contains SIG (exact
substring at line start after indentation), with its leading doc comments / attributes, through
its closing brace (or `;` for a braceless item). Prints what it removed."""
import sys


def item_end(s, i):
    """Index just past the item starting at s[i]: first `{` (balanced) or `;` at depth 0."""
    depth = 0
    n = len(s)
    j = i
    seen_brace = False
    while j < n:
        c = s[j]
        if c == '/' and s.startswith('//', j):
            j = s.index('\n', j) if '\n' in s[j:] else n
            continue
        if c == '/' and s.startswith('/*', j):
            j = s.index('*/', j) + 2
            continue
        if c == 'r' and (s.startswith('r"', j) or s.startswith('r#', j)) and not s[j - 1].isalnum() and s[j - 1] != '_':
            k = j + 1
            hashes = 0
            while s[k] == '#':
                hashes += 1
                k += 1
            if s[k] == '"':
                end = '"' + '#' * hashes
                j = s.index(end, k + 1) + len(end)
                continue
        if c == '"':
            k = j + 1
            while s[k] != '"':
                k += 2 if s[k] == '\\' else 1
            j = k + 1
            continue
        if c == "'":
            # char literal ('x', '\n', '\u{..}') vs lifetime ('a)
            if s[j + 1] == '\\':
                j = s.index("'", j + 2) + 1
                continue
            if j + 2 < n and s[j + 2] == "'":
                j += 3
                continue
        if c == '{' and depth == 0:
            # The body: from here only braces count.
            seen_brace = True
        if seen_brace:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return j + 1
        else:
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c == ';' and depth == 0:
                return j + 1
        j += 1
    raise ValueError('unterminated item')


def remove(s, sig):
    lines = s.split('\n')
    starts = [i for i, l in enumerate(lines) if l.lstrip().startswith(sig)]
    if len(starts) != 1:
        raise ValueError(f'{sig!r}: {len(starts)} matches')
    li = starts[0]
    # extend upward over doc comments / attributes
    top = li
    while top > 0 and (lines[top - 1].lstrip().startswith('///') or lines[top - 1].lstrip().startswith('#[')
                       or lines[top - 1].lstrip().startswith('//!') is False and lines[top - 1].lstrip().startswith('// ') and False):
        top -= 1
    off_top = sum(len(l) + 1 for l in lines[:top])
    off_sig = sum(len(l) + 1 for l in lines[:li])
    end = item_end(s, off_sig)
    # swallow the rest of the line and one blank line after
    if s[end:end + 1] == '\n':
        end += 1
    if s[end:end + 1] == '\n' and (off_top == 0 or s[off_top - 2:off_top] == '\n\n'):
        end += 1
    removed = s[off_top:end]
    print(f'removed {sig!r}: {removed.count(chr(10))} lines')
    return s[:off_top] + s[end:]


if __name__ == '__main__':
    f = sys.argv[1]
    s = open(f).read()
    for sig in sys.argv[2:]:
        s = remove(s, sig)
    open(f, 'w').write(s)
