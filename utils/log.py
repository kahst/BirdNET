from __future__ import print_function
import sys

log = ''
def show(s, new_line=True, discard=False):

    global log

    if isinstance(s, (list, tuple)):
        for i in range(len(s)):
            print(s[i], end=' '),
            if not discard:
                log += str(s[i])
                if i < len(s) - 1:
                    log += ' '
    else:
        print(s, end=' '),
        if not discard:
            log += str(s)

    if new_line:
        print('')
        if not discard:
            log += '\n'
    else:
        if not discard:
            log += ' '
    
    sys.stdout.flush()
    

def p(s, new_line=True, discard=False):

    show(s, new_line)

def clear():

    global log
    log = ''

def export(fpath):

    with open(fpath, 'w') as lfile:
        lfile.write(log)
    
