"""
API for the command-line I{pyflakes} tool.
"""


from __future__ import with_statement

import sys
import os
import _ast

### from pyflakes import checker, __version__
import checker # EKR
__version__ = 'EKR: __version__'
### from pyflakes import reporter as modReporter
import reporter as modReporter # EKR

__all__ = ['check', 'checkPath', 'checkRecursive', 'iterSourceCode', 'main']
def check(codeString, filename, reporter=None, tree=None): # EKR: added tree.
    """
    Check the Python source given by C{codeString} for flakes.

    @param codeString: The Python source to check.
    @type codeString: C{str}

    @param filename: The name of the file the source came from, used to report
        errors.
    @type filename: C{str}

    @param reporter: A L{Reporter} instance, where errors and warnings will be
        reported.

    @return: The number of warnings emitted.
    @rtype: C{int}
    """
    # print('EKR: api.check: %s' % filename)
    if reporter is None:
        reporter = modReporter._makeDefaultReporter()
    # First, compile into an AST and handle syntax errors.
    if not tree:
        try:
            tree = compile(codeString, filename, "exec", _ast.PyCF_ONLY_AST)
        except SyntaxError:
            value = sys.exc_info()[1]
            msg = value.args[0]
    
            (lineno, offset, text) = value.lineno, value.offset, value.text
            g.es_print_exception() # EKR
    
            # If there's an encoding problem with the file, the text is None.
            if text is None:
                # Avoid using msg, since for the only known case, it contains a
                # bogus message that claims the encoding the file declared was
                # unknown.
                # EKR:
                reporter.unexpectedError(filename, 'no text: syntax error?')
            else:
                reporter.syntaxError(filename, msg, lineno, offset, text)
            return 1
        except Exception:
            g.es_print_exception() # EKR
            reporter.unexpectedError(filename, 'unexpected exception')
            return 1
    # Okay, it's syntactically valid.  Now check it.
    w = checker.Checker(tree, filename)
    w.messages.sort(key=lambda m: m.lineno)
    for warning in w.messages:
        reporter.flake(warning)
    return len(w.messages)
def checkPath(filename, reporter=None):
    """
    Check the given path, printing out any warnings detected.

    @param reporter: A L{Reporter} instance, where errors and warnings will be
        reported.

    @return: the number of warnings printed
    """
    if reporter is None:
        reporter = modReporter._makeDefaultReporter()
    try:
        # in Python 2.6, compile() will choke on \r\n line endings. In later
        # versions of python it's smarter, and we want binary mode to give
        # compile() the best opportunity to do the right thing WRT text
        # encodings.
        if sys.version_info < (2, 7):
            mode = 'rU'
        else:
            mode = 'rb'

        with open(filename, mode) as f:
            codestr = f.read()
        if sys.version_info < (2, 7):
            codestr += '\n'     # Work around for Python <= 2.6
    except UnicodeError:
        reporter.unexpectedError(filename, 'problem decoding source')
        return 1
    except IOError:
        msg = sys.exc_info()[1]
        reporter.unexpectedError(filename, msg.args[1])
        return 1
    return check(codestr, filename, reporter)
def iterSourceCode(paths):
    """
    Iterate over all Python source files in C{paths}.

    @param paths: A list of paths.  Directories will be recursed into and
        any .py files found will be yielded.  Any non-directories will be
        yielded as-is.
    """
    for path in paths:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith('.py'):
                        yield os.path.join(dirpath, filename)
        else:
            yield path
def checkRecursive(paths, reporter):
    """
    Recursively check all source files in C{paths}.

    @param paths: A list of paths to Python source files and directories
        containing Python source files.
    @param reporter: A L{Reporter} where all of the warnings and errors
        will be reported to.
    @return: The number of warnings found.
    """
    warnings = 0
    for sourcePath in iterSourceCode(paths):
        warnings += checkPath(sourcePath, reporter)
    return warnings
def _exitOnSignal(sigName, message):
    """Handles a signal with sys.exit.

    Some of these signals (SIGPIPE, for example) don't exist or are invalid on
    Windows. So, ignore errors that might arise.
    """
    import signal

    try:
        sigNumber = getattr(signal, sigName)
    except AttributeError:
        # the signal constants defined in the signal module are defined by
        # whether the C library supports them or not. So, SIGPIPE might not
        # even be defined.
        return

    def handler(sig, f):
        sys.exit(message)

    try:
        signal.signal(sigNumber, handler)
    except ValueError:
        # It's also possible the signal is defined, but then it's invalid. In
        # this case, signal.signal raises ValueError.
        pass
def main(prog=None):
    """Entry point for the script "pyflakes"."""
    import optparse

    # Handle "Keyboard Interrupt" and "Broken pipe" gracefully
    _exitOnSignal('SIGINT', '... stopped')
    _exitOnSignal('SIGPIPE', 1)

    parser = optparse.OptionParser(prog=prog, version=__version__)
    (__, args) = parser.parse_args()
    reporter = modReporter._makeDefaultReporter()
    if args:
        warnings = checkRecursive(args, reporter)
    else:
        warnings = check(sys.stdin.read(), '<stdin>', reporter)
    raise SystemExit(warnings > 0)
