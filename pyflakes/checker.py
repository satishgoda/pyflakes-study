"""
Main module.

Implement the central Checker class.
Also, it models the Bindings and Scopes.
"""


# checker.py imports...
import leo.core.leoGlobals as g # EKR
assert g

import doctest
import os
import sys
import time # ekr

PY2 = sys.version_info < (3, 0)
PY32 = sys.version_info < (3, 3)    # Python 2.5 to 3.2
PY33 = sys.version_info < (3, 4)    # Python 2.5 to 3.3
builtin_vars = dir(__import__('__builtin__' if PY2 else 'builtins'))

try:
    import ast
except ImportError:     # Python 2.5
    import _ast as ast

    if 'decorator_list' not in ast.ClassDef._fields:
        # Patch the missing attribute 'decorator_list'
        ast.ClassDef.decorator_list = ()
        ast.FunctionDef.decorator_list = property(lambda s: s.decorators)

from pyflakes import messages
# g.trace(messages)

# Stats:
# jit:  not significantly better than original.
# aft:  checker: 1.97 total: 3.84 # 20% overall improvement.
# None: checker: 2.44 total: 4.99

aft = True
    # True: use AstFullTraverser class for traversals.
    # This is proving to be difficult, because ast.visit doesn't have a parent arg.
# jit = False
    # This is only slightly faster than the default handleChildren method.
    # True: create node handlers in getNodeHandler.
new_scope = aft
    # True: replace self.scope property by simple code in push/popStack.
stats = {}
    # Timing stats.
n_pass_nodes = [None, 0, 0]
    # Only Passes 1 & 2 traverse nodes.
    # The sum is the number of calls to handleNodes
n_ignore = n_handleChildren = n_FunctionDef = 0
n_load = n_store = n_scopes = 0
n_deferred_assignments = n_scope_names = 0
test_scope = None

# Globally defined names which are not attributes of the builtins module, or
# are only present on some platforms.
_MAGIC_GLOBALS = ['__file__', '__builtins__', 'WindowsError']

if aft:
    pass
else:
    

    class _FieldsOrder(dict):
        """Fix order of AST node fields."""

        def _get_fields(self, node_class):
            # handle iter before target, and generators before element
            # EKR: the effect of the key is to assign 0 to 'iter' or 'generators' or 'value'
            # and -1 to everything else. So the target is *last*, so reverse=True is needed.
            if 1: # EKR new code
                fields = list(node_class._fields)
                for field in ('iter', 'generators', 'value'):
                    if field in fields:
                        fields.remove(field)
                        fields.insert(0, field)
                        break
                return tuple(fields)
            else:
                fields = node_class._fields
                if 'iter' in fields:
                    key_first = 'iter'.find
                elif 'generators' in fields:
                    key_first = 'generators'.find
                else:
                    key_first = 'value'.find
                return tuple(sorted(fields, key=key_first, reverse=True))
            

        def __missing__(self, node_class):
            # EKR: called if self[node_class] does not exist.
            self[node_class] = fields = self._get_fields(node_class)
            # g.trace(node_class.__name__, fields)
            return fields

# Python >= 3.3 uses ast.Try instead of (ast.TryExcept + ast.TryFinally)
# EKR: used only by differentForks
if PY32:
    def getAlternatives(n):
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
else:
    def getAlternatives(n):
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]

def getNodeName(node):
    # Return the node's name, or None.
    return getattr(node, 'id', None) or getattr(node, 'name', None)
   

if PY2:
    def getNodeType(node_class):
        # workaround str.upper() which is locale-dependent
        # return str(unicode(node_class.__name__).upper())
        return node_class.__name__.upper()
            # EKR: hehe: pyflakes complains about unicode.
else:
    def getNodeType(node_class):
        return node_class.__name__.upper()

if aft:
    pass
else:

    def iter_child_nodes(node, omit=None, _fields_order=_FieldsOrder()):
        """
        Yield all direct child nodes of *node*, that is, all fields that
        are nodes and all items of fields that are lists of nodes.
        """
        for name in _fields_order[node.__class__]:
            if name == omit:
                continue
            field = getattr(node, name, None)
            if isinstance(field, ast.AST):
                yield field
            elif isinstance(field, list):
                for item in field:
                    yield item

def unit_test(raise_on_fail=True):
    '''Run basic unit tests for this file.'''
    import _ast
    # import leo.core.leoAst as leoAst
    # Compute all fields to test.
    aList = sorted(dir(_ast))
    remove = [
        'Interactive', 'Suite', # Not necessary.
        'PyCF_ONLY_AST', # A constant,
        'AST', # The base class,
        # Grammar symbols...
        'expr', 'mod', 'stmt',
        'boolop', 'cmpop', 'unaryop', 'operator',
        # Field names...
        'expr_context', 'excepthandler', 'slice', 'withitem',
    ]
    aList = [z for z in aList if not z.startswith('_') and not z in remove]
    # Now test them.
    # Create a real tree so handleNode doesn't have to test for an empty tree.
    fn, s = '<test>', 'pass'
    tree = compile(s, fn, "exec", ast.PyCF_ONLY_AST)
    ft = Checker(tree)
    operator_classes = (
        ast.cmpop, ast.boolop,
        ast.expr_context,
        ast.operator, ast.unaryop,
    )
    errors, nodes, operators = 0, 0, 0
    if aft:
        for z in aList:
            class_ = getattr(ast, z, None)
            if hasattr(ft, z):
                nodes += 1
            elif aft and issubclass(class_, operator_classes):
                operators += 1
            else:
                errors += 1
                print('Missing pyflakes visitor for: %s' % z)
    s = '%s node types, %s errors' % (nodes, errors)
    if raise_on_fail:
        assert not errors, s
    else:
        print(s)

# Binding and Definitions classes...


class Binding(object):
    """
    Represents the binding of a value to a name.

    The checker uses this to keep track of which names have been bound and
    which names have not. See L{Assignment} for a special type of binding that
    is checked with stricter rules.

    @ivar used: pair of (L{Scope}, line-number) indicating the scope and
                line number that this binding was last used
    """
    kind = 'binding'
    def __init__(self, name, source):
        self.name = name
        self.source = source # EKR: a node.
        self.used = False # EKR: Set in helpers of Name.


    def __str__(self):
        return self.name

    def __repr__(self):
        return '<Binding line %-2s %-6s %15s>' % (
            self.source.lineno,
            self.kind, 
            self.name,
        )
        # return '<%s object %r from line %r at 0x%x>' % (
            # self.__class__.__name__,
            # self.name,
            # self.source.lineno,
            # id(self))

    def redefines(self, other):
        return isinstance(other, Definition) and self.name == other.name


class Definition(Binding):
    """
    A binding that defines a function or a class.
    """


class Importation(Definition):
    """
    A binding created by an import statement.

    @ivar fullName: The complete name given to the import statement,
        possibly including multiple dotted components.
    @type fullName: C{str}
    """
    kind = 'import'

    def __init__(self, name, source):
        self.fullName = name
        self.redefined = []
        name = name.split('.')[0]
        super(Importation, self).__init__(name, source)

    def redefines(self, other):
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        else:
            # EKR: same as Binding.redefines.
            return isinstance(other, Definition) and self.name == other.name


class Argument(Binding):
    """
    Represents binding a name as an argument.
    """
    kind = 'arg' # EKR


class Assignment(Binding):
    """
    Represents binding a name with an explicit assignment.

    The checker will raise warnings for any Assignment that isn't used. Also,
    the checker does not consider assignments in tuple/list unpacking to be
    Assignments, rather it treats them as simple Bindings.
    """
    kind = 'assign' # EKR


class FunctionDefinition(Definition):
    
    kind = 'def' # EKR


class ClassDefinition(Definition):
    
    kind = 'class' # EKR


class ExportBinding(Binding):
    """
    A binding created by an C{__all__} assignment.  If the names in the list
    can be determined statically, they will be treated as names for export and
    additional checking applied to them.

    The only C{__all__} assignment that can be recognized is one which takes
    the value of a literal list containing literal strings.  For example::

        __all__ = ["foo", "bar"]

    Names which are imported and not otherwise used but appear in the value of
    C{__all__} will not have an unused import warning reported for them.
    """
    kind = 'export' # EKR
    

    def __init__(self, name, source, scope):
        if '__all__' in scope and isinstance(source, ast.AugAssign):
            self.names = list(scope['__all__'].names)
        else:
            self.names = []
        self.kind = 'import'
        if isinstance(source.value, (ast.List, ast.Tuple)):
            for node in source.value.elts:
                if isinstance(node, ast.Str):
                    self.names.append(node.s)
        super(ExportBinding, self).__init__(name, source)

# Scope classes...


class Scope(dict):

    importStarred = False
        # set to True when import * is found

    # EKR: Adding more data to scopes takes negligible time.
    def __init__(self, node, name, parent):
        self.name = name
        self.node = node
        self.parent = parent
        # This code would be useful in some contexts, but not here and now.
        # self.children = []
        # if self.parent:
            # self.parent.children.append(self)
        

    def __repr__(self):
        # scope_class = self.__class__.__name__
        # return '<%s at 0x%x %s>' % (scope_class, id(self), dict.__repr__(self))
        # return '%s %s' % (scope_class, self.name)
        return self.name

    


class ClassScope(Scope):

    def __init__(self, node, name, parent):
        Scope.__init__(self, node, name, parent)



class FunctionScope(Scope):
    """
    I represent a name scope for a function.

    @ivar globals: Names declared 'global' in this function.
    """
    # EKR: only FunctionScope defines .globals ivar.
    usesLocals = False
    alwaysUsed = set([
        '__tracebackhide__',
        '__traceback_info__',
        '__traceback_supplement__'])


    def __init__(self, node, name, parent):

        Scope.__init__(self, node, name, parent)
        # Simplify: manage the special locals as globals
        self.globals = self.alwaysUsed.copy()
        self.returnValue = None     # First non-empty return
        self.isGenerator = False    # Detect a generator

    def unusedAssignments(self):
        """
        Return a generator for the assignments which have not been used.
        """
        # EKR: only called in FunctionScope.
        global n_scope_names ; n_scope_names += len(self.keys())
        for name, binding in self.items():
            if (not binding.used and
                name not in self.globals and
                not self.usesLocals and
                isinstance(binding, Assignment)
            ):
                yield name, binding


class GeneratorScope(Scope):

    def __init__(self, node, name, parent):
        Scope.__init__(self, node, name, parent)


class ModuleScope(Scope):

    def __init__(self, node, name, parent):
        assert parent is None, parent
            # Module's are the only scopes without a parent.
        name = 'Module: %s' % name
        Scope.__init__(self, node, name, parent)


class Checker(object):
    """
    I check the cleanliness and sanity of Python code.

    @ivar _deferredFunctions: Tracking list used by L{deferFunction}.  Elements
        of the list are two-tuples.  The first element is the callable passed
        to L{deferFunction}.  The second element is a copy of the scope stack
        at the time L{deferFunction} was called.

    @ivar _deferredAssignments: Similar to C{_deferredFunctions}, but for
        callables which are deferred assignment checks.
    """

    nodeDepth = 0 # EKR: also set in ctor.
    offset = None
    traceTree = False

    builtIns = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns = os.environ.get('PYFLAKES_BUILTINS')
    if _customBuiltIns:
        builtIns.update(_customBuiltIns.split(','))
    del _customBuiltIns


    def __init__(self, tree, filename='(none)', builtins=None,
                 withDoctest='PYFLAKES_DOCTEST' in os.environ):
        
        if aft:
            pass
        else:
            self._nodeHandlers = {}
        self._deferredFunctions = []
        self._deferredAssignments = []
        self.deadScopes = []
        self.messages = []
        self.nodeDepth = 0
        self.filename = filename
        # EKR: self.builtIns defined in class node.
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest = withDoctest
        self.exceptHandlers = [()]
        self.futuresAllowed = True
        self.root = tree
        self.pass_n = 1 # EKR.
        if new_scope: # EKR.
            self.scope = None
        self.handleNode(tree, parent=None)
            # EKR: new MODULE handler does all the work.

    def deferAssignment(self, func):
        """
        Schedule an assignment handler to be called just after deferred
        function handlers.
        """
        global n_deferred_assignments ; n_deferred_assignments += 1
        self._deferredAssignments.append((func, self.scopeStack[:], self.offset))

    def deferFunction(self, func, node=None, args=None):
        """
        Schedule a function handler to be called just before completion.

        This is used for handling function bodies, which must be deferred
        because code later in the file might modify the global scope. When
        `callable` is called, the scope at the time this is called will be
        restored, however it will contain any new bindings added to it.
        """
        self._deferredFunctions.append((func, self.scopeStack[:], self.offset))

    def runDeferred(self, deferred):
        """
        Run the callables in C{deferred} using their associated scope stack.
        """
        for handler, scope, offset in deferred:
            self.scopeStack = scope
            self.offset = offset
            if new_scope:
                self.scope = self.scopeStack[-1] if self.scopeStack else None
            handler()

    if new_scope:
        pass
    else:
        # EKR: expensive.
        # It's not necessary, because pushStack/popStack encapsulate it.
        @property
        def scope(self):
            if new_scope: assert False
            return self.scopeStack[-1]

    if new_scope:
        
        def popScope(self):
            self.deadScopes.append(self.scopeStack.pop())
            self.scope = self.scopeStack[-1] if self.scopeStack else None

    else:
        
        def popScope(self):
            self.deadScopes.append(self.scopeStack.pop())

    if new_scope:
        
        def pushScope(self, node, name, scopeClass):
            global n_scopes ; n_scopes += 1
            parent = self.scopeStack and self.scopeStack[-1] or None
            self.scope = scopeClass(node, name, parent)
            self.scopeStack.append(self.scope)
            
    else:
                
        def pushScope(self, node, name, scopeClass):
            global n_scopes ; n_scopes += 1
            parent = self.scopeStack and self.scopeStack[-1] or None
            scope = scopeClass(node, name, parent)
            self.scopeStack.append(scope)


    def report(self, messageClass, *args, **kwargs):
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node):
        # Lookup the first parent which is not Tuple, List or Starred
        # EKR: handleNode sets node.parent.
        while True:
            node = node.parent
            if not hasattr(node, 'elts') and not hasattr(node, 'ctx'):
                return node

    def getCommonAncestor(self, lnode, rnode, stop):
        if stop in (lnode, rnode) or not (hasattr(lnode, 'parent') and
                                          hasattr(rnode, 'parent')):
            return None
        if lnode is rnode:
            return lnode

        if (lnode.depth > rnode.depth):
            return self.getCommonAncestor(lnode.parent, rnode, stop)
        if (lnode.depth < rnode.depth):
            return self.getCommonAncestor(lnode, rnode.parent, stop)
        return self.getCommonAncestor(lnode.parent, rnode.parent, stop)

    def descendantOf(self, node, ancestors, stop):
        for a in ancestors:
            if self.getCommonAncestor(node, a, stop):
                return True
        return False

    def differentForks(self, lnode, rnode):
        """True, if lnode and rnode are located on different forks of IF/TRY"""
        ancestor = self.getCommonAncestor(lnode, rnode, self.root)
        parts = getAlternatives(ancestor)
        if parts:
            for items in parts:
                if self.descendantOf(lnode, items, ancestor) ^ \
                   self.descendantOf(rnode, items, ancestor):
                    return True
        return False

    def addBinding(self, node, value):
        """
        Called when a binding is altered.

        - `node` is the statement responsible for the change
        - `value` is the new value, a Binding instance
        """
        trace = False and test_scope == 'test'
        # assert value.source in (node, node.parent):
        for scope in self.scopeStack[::-1]:
                # EKR: same as list(reversed(scopeStack))
            if value.name in scope:
                break
        existing = scope.get(value.name)

        if existing and not self.differentForks(node, existing.source):

            parent_stmt = self.getParent(value.source)
            if isinstance(existing, Importation) and isinstance(parent_stmt, ast.For):
                self.report(messages.ImportShadowedByLoopVar,
                            node, value.name, existing.source)

            elif scope is self.scope:
                # g.trace('====', scope, existing, existing.used, value)
                if (isinstance(parent_stmt, ast.comprehension) and
                    not isinstance(self.getParent(existing.source),
                        (ast.For, ast.comprehension))
                ):
                    # g.trace(value.name, parent_stmt, self.getParent(existing.source))
                    # g.trace(g.callers(10))
                    self.report(messages.RedefinedInListComp,
                                node, value.name, existing.source)
                elif not existing.used and value.redefines(existing):
                    # Redefines Class or Function.
                    self.report(messages.RedefinedWhileUnused,
                                node, value.name, existing.source)

            elif isinstance(existing, Importation) and value.redefines(existing):
                existing.redefined.append(node)

        if value.name in self.scope:
            # then assume the rebound name is used as a global or within a loop
            value.used = self.scope[value.name].used

        self.scope[value.name] = value
        if trace: g.trace('    pass: %s %20r in %s' % (
            self.pass_n, value, scope.name))
            # getattr(self.scope, 'name', self.scope.__class__.__name__)))

    def isDocstring(self, node):
        """
        Determine if the given node is a docstring, as long as it is at the
        correct place in the node tree.
        """
        return isinstance(node, ast.Str) or (isinstance(node, ast.Expr) and
                                             isinstance(node.value, ast.Str))

    def getDocstring(self, node):
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Str):
            return (None, None)
        # Computed incorrectly if the docstring has backslash
        doctest_lineno = node.lineno - node.s.count('\n') - 1
        return (node.s, doctest_lineno)

    ignore_kinds = {}

    def ignore(self, node):
        
        # EKR: Ignoring a node is not strictly the same as not calling handleNode
        # because handleNode sets node.parent and node.depth fields.
        # However, these fields aren't used for ignored nodes.
        
        global n_ignore ; n_ignore += 1
        if aft:
            name = node.__class__.__name__
            if name not in self.ignore_kinds:
                self.ignore_kinds[name] = True
                g.trace(name, g.callers())


    def getNodeHandler(self, node_class):

        try:
            return self._nodeHandlers[node_class]
        except KeyError:
            nodeType = getNodeType(node_class)
        self._nodeHandlers[node_class] = handler = getattr(self, nodeType)
        return handler

    def handleChildren(self, tree, omit=None):
        # EKR: iter_child_nodes uses _FieldsOrder class.
        global n_handleChildren ; n_handleChildren += 1
        
        assert not aft, g.callers()
        
        for node in iter_child_nodes(tree, omit=omit):
            self.handleNode(node, tree)
    if aft:

        if aft:

            def alias(self, node):
                pass

            # 2: arguments = (expr* args, identifier? vararg,
            #                 identifier? kwarg, expr* defaults)
            # 3: arguments = (arg*  args, arg? vararg,
            #                 arg* kwonlyargs, expr* kw_defaults,
            #                 arg? kwarg, expr* defaults)

            def arguments(self, node):

                for z in node.args:
                    self.handleNode(z, node)
                if g.isPython3 and getattr(node, 'vararg', None):
                    # An identifier in Python 2.
                    self.handleNode(node.vararg, node)
                if getattr(node, 'kwonlyargs', None): # Python 3.
                    assert isinstance(aList, (list, tuple)), repr(aList)
                    for z in aList:
                        self.handleNode(z, node)
                if getattr(node, 'kw_defaults', None): # Python 3.
                    assert isinstance(aList, (list, tuple)), repr(aList)
                    for z in aList:
                        self.handleNode(z, node)
                if g.isPython3 and getattr(node, 'kwarg', None):
                    # An identifier in Python 2.
                    self.handleNode(node.kwarg, node)
                for z in node.defaults:
                    self.handleNode(z, node)

            # 3: arg = (identifier arg, expr? annotation)

            def arg(self, node):
                if getattr(node, 'annotation', None):
                    self.handleNode(node.annotation, node)

            # Attribute(expr value, identifier attr, expr_context ctx)

            def Attribute(self, node):
                self.handleNode(node.value, node)
                # self.handleNode(node.ctx, node)

            # BinOp(expr left, operator op, expr right)

            def BinOp(self, node):
                self.handleNode(node.left, node)
                # self.op_name(node.op)
                self.handleNode(node.right, node)

            # BoolOp(boolop op, expr* values)

            def BoolOp(self, node):
                
                # self.handleNode(node.op)
                for z in node.values:
                    self.handleNode(z, node)

            def Bytes(self, node):
                pass

            # Call(expr func, expr* args, keyword* keywords, expr? starargs, expr? kwargs)

            def Call(self, node):
                # Call the nodes in token order.
                self.handleNode(node.func, node)
                for z in node.args:
                    self.handleNode(z, node)
                for z in node.keywords:
                    self.handleNode(z, node)
                if getattr(node, 'starargs', None):
                    self.handleNode(node.starargs, node)
                if getattr(node, 'kwargs', None):
                    self.handleNode(node.kwargs, node)

            # Compare(expr left, cmpop* ops, expr* comparators)

            def Compare(self, node):
                # Visit all nodes in token order.
                self.handleNode(node.left, node)
                assert len(node.ops) == len(node.comparators)
                for i in range(len(node.ops)):
                    if not isinstance(node.ops[i], ast.cmpop):
                        # Could be a name, etc.
                        self.handleNode(node.ops[i], node)
                    if not isinstance(node.comparators[i], ast.cmpop):
                        self.handleNode(node.comparators[i], node)

            # comprehension (expr target, expr iter, expr* ifs)

            def comprehension(self, node):
                # EKR: visit iter first.
                self.handleNode(node.iter, node) # An attribute.
                self.handleNode(node.target, node) # A name.
                for z in node.ifs:
                    self.handleNode(z, node)

            # Dict(expr* keys, expr* values)

            def Dict(self, node):
                # Visit all nodes in token order.
                assert len(node.keys) == len(node.values)
                for i in range(len(node.keys)):
                    self.handleNode(node.keys[i], node)
                    self.handleNode(node.values[i], node)

            def Ellipsis(self, node):
                pass

            # Expr(expr value)

            def Expr(self, node):
                self.handleNode(node.value, node)

            def Expression(self, node):
                '''An inner expression'''
                self.handleNode(node.body, node)

            def ExtSlice(self, node):
                for z in node.dims:
                    self.handleNode(z, node)

            # IfExp(expr test, expr body, expr orelse)

            def IfExp(self, node):
                self.handleNode(node.body, node)
                self.handleNode(node.test, node)
                self.handleNode(node.orelse, node)

            def Index(self, node):
                self.handleNode(node.value, node)

            # keyword = (identifier arg, expr value)

            def keyword(self, node):
                # node.arg is a string.
                self.handleNode(node.value, node)

            # List(expr* elts, expr_context ctx)

            def List(self, node):
               
                for z in node.elts:
                    self.handleNode(z, node)
                # self.handleNode(node.ctx, node)

            # It would not be good to have to test for lists.
            # if aft:
                # def list(self, node):
                    # g.trace(g.callers())
                    # assert self.pass_n == 2, self.pass_n
                    # for z in node:
                        # self.handleNode(z, node)

            def NameConstant(self, node): # Python 3 only.

                assert aft
                assert isinstance(node.value, (bool, str, None.__class__)), node.value.__class__.__name__
                # g.trace(node.value)
                # if node.value:
                #     self.handleNode(node.value, node)
                # self.handleNode(node.value, node)
                # s = repr(node.value)
                # return 'bool' if s in ('True', 'False') else s

            def Num(self, node):
                pass

            # Python 2.x only
            # Repr(expr value)

            def Repr(self, node):
                self.handleNode(node.value, node)

            # Set(expr* elts)

            def Set(self, node):
                for z in node.elts:
                    self.handleNode(z, node)
                    

            def Slice(self, node):
                if getattr(node, 'lower', None):
                    self.handleNode(node.lower, node)
                if getattr(node, 'upper', None):
                    self.handleNode(node.upper, node)
                if getattr(node, 'step', None):
                    self.handleNode(node.step, node)

            def Str(self, node):
                pass

            # Subscript(expr value, slice slice, expr_context ctx)

            def Subscript(self, node):
                # EKR: Visit value first.
                self.handleNode(node.value, node)
                self.handleNode(node.slice, node)
                # self.handleNode(node.ctx, node)

            # Tuple(expr* elts, expr_context ctx)

            def Tuple(self, node):
                # g.trace(node.elts)
                for z in node.elts:
                    self.handleNode(z, node)
                # self.handleNode(node.ctx, node)
                    # EKR: ignore ctx fields (LOAD, STORE, etc.

            # UnaryOp(unaryop op, expr operand)

            def UnaryOp(self, node):
                # self.op_name(node.op)
                self.handleNode(node.operand, node)

            # Assert(expr test, expr? msg)

            def Assert(self, node):
                self.handleNode(node.test, node)
                if node.msg:
                    self.handleNode(node.msg, node)

            # Assign(expr* targets, expr value)

            def Assign(self, node):
                # EKR: Visit value first.
                self.handleNode(node.value, node)
                for z in node.targets:
                    self.handleNode(z, node)
                

            def Break(self, node):
                pass

            def Continue(self, node):
                pass

            # Delete(expr* targets)

            def Delete(self, node):
                for z in node.targets:
                    self.handleNode(z, node)

            # Python 2.x only
            # Exec(expr body, expr? globals, expr? locals)

            def Exec(self, node):
                self.handleNode(node.body, node)
                if getattr(node, 'globals', None):
                    self.handleNode(node.globals, node)
                if getattr(node, 'locals', None):
                    self.handleNode(node.locals, node)

            # For(expr target, expr iter, stmt* body, stmt* orelse)

            def For(self, node):
                
                # EKR: visit iter first.
                self.handleNode(node.iter, node)
                self.handleNode(node.target, node)
                for z in node.body:
                    self.handleNode(z, node)
                for z in node.orelse:
                    self.handleNode(z, node)

            AsyncFor = For

            # If(expr test, stmt* body, stmt* orelse)

            def If(self, node):

                if not isinstance(node.test, ast.operator):
                    self.handleNode(node.test, node)
                for z in node.body:
                    self.handleNode(z, node)
                for z in node.orelse:
                    self.handleNode(z, node)

            def Pass(self, node):
                pass

            # Python 2.x only
            # Print(expr? dest, expr* values, bool nl)

            def Print(self, node):
                if getattr(node, 'dest', None):
                    self.handleNode(node.dest, node)
                for expr in node.values:
                    self.handleNode(expr, node)

            # Raise(expr? type, expr? inst, expr? tback)    Python 3
            # Raise(expr? exc, expr? cause)                 Python 2

            def Raise(self, node):

                if g.isPython3:
                    if getattr(node, 'exc', None):
                        self.handleNode(node.exc, node)
                    if getattr(node, 'cause', None):
                        self.handleNode(node.cause, node)
                else:
                    if getattr(node, 'type', None):
                        self.handleNode(node.type, node)
                    if getattr(node, 'inst', None):
                        self.handleNode(node.inst, node)
                    if getattr(node, 'tback', None):
                        self.handleNode(node.tback, node)

            # Starred(expr value, expr_context ctx)

            def Starred(self, node):

                self.handleNode(node.value, node)

            # TryFinally(stmt* body, stmt* finalbody)

            def TryFinally(self, node):
                for z in node.body:
                    self.handleNode(z, node)
                for z in node.finalbody:
                    self.handleNode(z, node)

            # While(expr test, stmt* body, stmt* orelse)

            def While(self, node):
                self.handleNode(node.test, node) # Bug fix: 2013/03/23.
                for z in node.body:
                    self.handleNode(z, node)
                for z in node.orelse:
                    self.handleNode(z, node)

            # 2:  With(expr context_expr, expr? optional_vars,
            #          stmt* body)
            # 3:  With(withitem* items,
            #          stmt* body)
            # withitem = (expr context_expr, expr? optional_vars)

            def With(self, node):
                if getattr(node, 'context_expr', None):
                    self.handleNode(node.context_expr, node)
                if getattr(node, 'optional_vars', None):
                    self.handleNode(node.optional_vars, node)
                if getattr(node, 'items', None): # Python 3.
                    for item in node.items:
                        self.handleNode(item.context_expr, node)
                        if getattr(item, 'optional_vars', None):
                            try:
                                for z in item.optional_vars:
                                    self.handleNode(z, node)
                            except TypeError: # Not iterable.
                                self.handleNode(item.optional_vars, node)
                for z in node.body:
                    self.handleNode(z, node)
                    
            AsyncWith = With
    else:
        CONTINUE = BREAK = PASS = ignore
        NUM = STR = BYTES = ELLIPSIS = ignore

        # EKR: AstFullTraverser doesn't visit these nodes.
        # expression contexts are node instances too, though being constants
        LOAD = STORE = DEL = AUGLOAD = AUGSTORE = PARAM = ignore
        
        # EKR: AstFullTraverser doesn't visit these nodes.
        # same for operators
        AND = OR = ADD = SUB = MULT = DIV = MOD = POW = LSHIFT = RSHIFT = \
            BITOR = BITXOR = BITAND = FLOORDIV = INVERT = NOT = UADD = USUB = \
            EQ = NOTEQ = LT = LTE = GT = GTE = IS = ISNOT = IN = NOTIN = ignore
            
        # EKR: MatMult is new in Python 3.5
        MATMULT = ignore

        # "stmt" type nodes
        DELETE = PRINT = FOR = ASYNCFOR = WHILE = IF = WITH = WITHITEM = \
            ASYNCWITH = ASYNCWITHITEM = RAISE = TRYFINALLY = ASSERT = EXEC = \
            EXPR = ASSIGN = handleChildren
        
        # "expr" type nodes
        BOOLOP = BINOP = UNARYOP = IFEXP = DICT = SET = \
            COMPARE = CALL = REPR = ATTRIBUTE = SUBSCRIPT = LIST = TUPLE = \
            STARRED = NAMECONSTANT = handleChildren
        
        # "slice" type nodes
        SLICE = EXTSLICE = INDEX = handleChildren
        
        # additional node types
        COMPREHENSION = KEYWORD = handleChildren

    # EKR: like visitors

    def handleDoctests(self, node):

        try:
            (docstring, node_lineno) = self.getDocstring(node.body[0])
            examples = docstring and self._getDoctestExamples(docstring)
        except (ValueError, IndexError):
            # e.g. line 6 of the docstring for <string> has inconsistent
            # leading whitespace: ...
            return
        if not examples:
            return
        if 0:
            g.trace('=========', g.callers())
            g.trace('examples:', examples)
        node_offset = self.offset or (0, 0)
        name = getattr(node, 'name', None) # EKR
        self.pushScope(node, name, FunctionScope)
        if 0:
            print('')
            g.trace(self.offset, node, self.scopeStack)
        underscore_in_builtins = '_' in self.builtIns
        if not underscore_in_builtins:
            self.builtIns.add('_')
        for example in examples:
            try:
                tree = compile(example.source, "<doctest>", "exec", ast.PyCF_ONLY_AST)
            except SyntaxError:
                e = sys.exc_info()[1]
                position = (node_lineno + example.lineno + e.lineno,
                            example.indent + 4 + (e.offset or 0))
                self.report(messages.DoctestSyntaxError, node, position)
            else:
                self.offset = (node_offset[0] + node_lineno + example.lineno,
                               node_offset[1] + example.indent + 4)
                assert isinstance(tree, ast.Module)
                if aft:
                    # Explicitly visit the module's children.
                    for z in tree.body:
                        self.handleNode(z, tree)
                else:
                    self.handleChildren(tree)
                self.offset = node_offset
        if not underscore_in_builtins:
            self.builtIns.remove('_')
        self.popScope()

    def handleNode(self, node, parent):
        # EKR: this the general node visiter.
        # assert isinstance(node, (ast.AST, ast.AsyncWith)), repr(node)
        global n_pass_nodes
        assert node, g.callers()
        # The following will fail unless 0 < self.pass_n < 3
        n_pass_nodes[self.pass_n] += 1
        # EKR: used only when running doctests.
        if self.offset and getattr(node, 'lineno', None) is not None:
            node.lineno += self.offset[0]
            node.col_offset += self.offset[1]
        # if self.traceTree:
            # print('  ' * self.nodeDepth(node) + node.__class__.__name__)
        if (self.futuresAllowed and
            not (isinstance(node, (ast.Module, ast.ImportFrom)) or self.isDocstring(node))
                 # EKR: works regardless of new_module.
        ):
            self.futuresAllowed = False
        # EKR: getCommonAncestor uses node.depth.
        self.nodeDepth += 1
        node.depth = self.nodeDepth
        node.parent = parent
        if aft:
            handler = getattr(self, node.__class__.__name__)
            handler(node)
        else:
            # EKR: this is the only call to getNodeHandler.
            handler = self.getNodeHandler(node.__class__)
            handler(node)
        self.nodeDepth -= 1
        # if self.traceTree:
            # print('  ' * self.nodeDepth(node) + 'end ' + node.__class__.__name__)

    _getDoctestExamples = doctest.DocTestParser().get_examples

    def AUGASSIGN(self, node):
        self.handleNodeLoad(node.target)
        self.handleNode(node.value, node)
        self.handleNode(node.target, node)

    if aft:
        AugAssign = AUGASSIGN

    def CLASSDEF(self, node):
        """
        Check names used in a class definition, including its decorators, base
        classes, and the body of its definition.  Additionally, add its name to
        the current scope.
        """
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        for baseNode in node.bases:
            self.handleNode(baseNode, node)
        if not PY2:
            for keywordNode in node.keywords:
                self.handleNode(keywordNode, node)
        self.pushScope(node, node.name, ClassScope)
        if self.withDoctest:
            self.deferFunction(lambda: self.handleDoctests(node))
        # EKR: Unlike def's & lambda's, we *do* traverse the class's body.
        for stmt in node.body:
            self.handleNode(stmt, node)
        self.popScope()
        self.addBinding(node, ClassDefinition(node.name, node))
        
    if aft:
        ClassDef = CLASSDEF

    # Python 2: ExceptHandler(expr? type, expr? name, stmt* body)
    # Python 3: ExceptHandler(expr? type, identifier? name, stmt* body)

    def EXCEPTHANDLER(self, node):
        # 3.x: in addition to handling children, we must handle the name of
        # the exception, which is not a Name node, but a simple string.
        if aft:
            if g.isPython3:
                if isinstance(node.name, str):
                    self.handleNodeStore(node)
            elif node.name:
                self.handleNode(node.name, node)
            if node.type:
                self.handleNode(node.type, node)
            for z in node.body:
                self.handleNode(z, node)
        else:
            if isinstance(node.name, str):
                self.handleNodeStore(node)  
            self.handleChildren(node)
        
    if aft:
        ExceptHandler = EXCEPTHANDLER


    def FUNCTIONDEF(self, node):
        
        global n_FunctionDef ; n_FunctionDef += 1
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        self.LAMBDA(node) # EKR: defer's traversal of the body!
        self.addBinding(node, FunctionDefinition(node.name, node))
        if self.withDoctest:
            self.deferFunction(lambda: self.handleDoctests(node))
    if aft:
        FunctionDef = AsyncFunctionDef = FUNCTIONDEF

    ASYNCFUNCTIONDEF = FUNCTIONDEF

    # GeneratorExp(expr elt, comprehension* generators)
    # SetComp(expr elt, comprehension* generators)

    def GENERATOREXP(self, node):
        # EKR: always push a new scope.
        name = 'Generator: %s' % id(node)
        self.pushScope(node, name, GeneratorScope)
        if aft:
            # EKR: call generators first.
            for z in node.generators:
                self.handleNode(z, node)
            self.handleNode(node.elt, node)
        else:
            self.handleChildren(node)
        self.popScope()
        
    if aft:
        SetComp = GeneratorExp = GENERATOREXP
        
    # DictComp(expr key, expr value, comprehension* generators)
        
    def DictComp(self, node):
        name = 'Generator: %s' % id(node)
        self.pushScope(node, name, GeneratorScope)
        if aft:
            # EKR: call generators first.
            for z in node.generators:
                self.handleNode(z, node)
            self.handleNode(node.key, node)
            self.handleNode(node.value, node)
        else:
            self.handleChildren(node)
        self.popScope()
        
    def ListComp(self, node):
        # EKR: Push a new scope only in Python 3.
        name = 'Generator: %s' % id(node)
        if g.isPython3:
            self.pushScope(node, name, GeneratorScope)
        if aft:
            # EKR: call generators first.
            for z in node.generators:
                self.handleNode(z, node)
            self.handleNode(node.elt, node)
        else:
            self.handleChildren(node)
        if g.isPython3:
            self.popScope()

    LISTCOMP = handleChildren if PY2 else GENERATOREXP
        
    DICTCOMP = SETCOMP = GENERATOREXP

    def GLOBAL(self, node):
        """
        Keep track of globals declarations.
        """
        # In doctests, the global scope is an anonymous function at index 1.
        global_scope_index = 1 if self.withDoctest else 0
        global_scope = self.scopeStack[global_scope_index]

        # Ignore 'global' statement in global scope.
        if self.scope is not global_scope:

            # One 'global' statement can bind multiple (comma-delimited) names.
            for node_name in node.names:
                node_value = Assignment(node_name, node)

                # Remove UndefinedName messages already reported for this name.
                self.messages = [
                    m for m in self.messages if
                        not isinstance(m, messages.UndefinedName) and
                        not isinstance(m, messages.ReturnOutsideFunction) and
                            # EKR: Real bug fix.
                        m.message_args[0] != node_name]

                # Bind name to global scope if it doesn't exist already.
                global_scope.setdefault(node_name, node_value)

                # Bind name to non-global scopes, but as already "used".
                node_value.used = (global_scope, node)
                for scope in self.scopeStack[global_scope_index + 1:]:
                    scope[node_name] = node_value

    if aft:
        Global = Nonlocal = GLOBAL

    NONLOCAL = GLOBAL

    def IMPORT(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            importation = Importation(name, node)
            self.addBinding(node, importation)

    if aft:
        Import = IMPORT

    def IMPORTFROM(self, node):
        if node.module == '__future__':
            if not self.futuresAllowed:
                self.report(messages.LateFutureImport,
                            node, [n.name for n in node.names])
        else:
            self.futuresAllowed = False

        for alias in node.names:
            if alias.name == '*':
                self.scope.importStarred = True
                self.report(messages.ImportStarUsed, node, node.module)
                continue
            name = alias.asname or alias.name
            importation = Importation(name, node)
            if node.module == '__future__':
                importation.used = (self.scope, node)
            self.addBinding(node, importation)
            
    if aft:
        ImportFrom = IMPORTFROM


    def LAMBDA(self, node):
        
        # Pass 1: visit *only* annotations and defaults.
        annotations, args, defaults = self.get_function_args(node)
        for child in annotations + defaults:
            if child:
                self.handleNode(child, node)
        # EKR: The dog that isn't barking:
        # pass 1 defers traversing the def's/lambda's body!

        def runFunction():
            '''A function that will be run in pass 2.'''

            assert self.pass_n == 2, self.pass_n
            if hasattr(node, 'name'):
                def_name = 'def: %s' % node.name
            else:
                def_name = 'Lambda: %s' % id(node)
            self.pushScope(node, def_name, FunctionScope)
            for name in args:
                self.addBinding(node, Argument(name, node))
                
            # EKR: *Now* traverse the body of the Def/Lambda.
            if isinstance(node.body, list):
                # case for FunctionDefs
                for stmt in node.body:
                    self.handleNode(stmt, node)
            else:
                # case for Lambdas
                self.handleNode(node.body, node)
                
            # EKR: defer checking assignments until pass 3.

            def checkUnusedAssignments():
                """
                Check to see if any assignments have not been used.
                """
                assert self.pass_n == 3, self.pass_n
                # g.trace(self.scope)
                for name, binding in self.scope.unusedAssignments():
                    self.report(messages.UnusedVariable, binding.source, name)
            
            self.deferAssignment(checkUnusedAssignments)

            if PY32:
                def checkReturnWithArgumentInsideGenerator():
                    """
                    Check to see if there is any return statement with
                    arguments but the function is a generator.
                    """
                    # g.trace(self.scope)
                    assert self.pass_n == 3, self.pass_n
                    if self.scope.isGenerator and self.scope.returnValue:
                        self.report(messages.ReturnWithArgsInsideGenerator,
                                    self.scope.returnValue)
                self.deferAssignment(checkReturnWithArgumentInsideGenerator)

            self.popScope()
        self.deferFunction(runFunction)

    if aft:
        Lambda = LAMBDA

    def get_function_args(self, node):

        annotations, args = [], []

        if PY2:
            def addArgs(arglist):
                for arg in arglist:
                    if isinstance(arg, ast.Tuple):
                        addArgs(arg.elts)
                    else:
                        args.append(arg.id)
            addArgs(node.args.args)
            defaults = node.args.defaults
        else:
            for arg in node.args.args + node.args.kwonlyargs:
                args.append(arg.arg)
                annotations.append(arg.annotation)
            defaults = node.args.defaults + node.args.kw_defaults

        # Only for Python3 FunctionDefs
        is_py3_func = hasattr(node, 'returns')

        for arg_name in ('vararg', 'kwarg'):
            wildcard = getattr(node.args, arg_name)
            if not wildcard:
                continue
            args.append(wildcard if PY33 else wildcard.arg)
            if is_py3_func:
                if PY33:  # Python 2.5 to 3.3
                    argannotation = arg_name + 'annotation'
                    annotations.append(getattr(node.args, argannotation))
                else:     # Python >= 3.4
                    annotations.append(wildcard.annotation)

        if is_py3_func:
            annotations.append(node.returns)

        if len(set(args)) < len(args):
            for (idx, arg) in enumerate(args):
                if arg in args[:idx]:
                    self.report(messages.DuplicateArgument, node, arg)
                    
        return annotations, args, defaults

    def MODULE(self, node):
        global stats
        if new_scope:
            self.scopeStack = []
            self.pushScope(node, self.filename, ModuleScope)
        else:
            self.scopeStack = [ModuleScope(node, self.filename, None)]
        moduleScope = self.scopeStack[-1] # EKR: I think I introduced this.
        self.pass1(node)
            # Traverse all top-level symbols.
        self.pass2(node)
            # Traverse all def/lambda bodies.
        self.pass3(node)
            # Check deferred assignments.
        if new_scope:
            self.scopeStack = []
            self.scope = None
        else:
            del self.scopeStack[1:]
        self.deadScopes.append(moduleScope)
        self.checkDeadScopes()
            # Pass 4.

    if aft:
        Module = MODULE

    def pass1(self, node):
        
        global stats
        t1 = time.clock()
        self.pass_n = 1
        # This looks like it is a full pass.
        # In fact, traversing of def/lambda happens in pass 2.
        if aft:
            for z in node.body:
                self.handleNode(z, node)
        else:
            self.handleChildren(node)
        t2 = time.clock()
        stats['pass1'] = stats.get('pass1', 0.0) + t2-t1

    def pass2(self, node):
        
        global stats
        t1 = time.clock()
        # Traverse the bodies of all def's & lambda's.
        self.pass_n = 2
        self.runDeferred(self._deferredFunctions)
            # Run all the queued runFunction functions or scanFunction methods.
        self._deferredFunctions = None
            # Set _deferredFunctions to None so that deferFunction will fail
            # noisily if called after we've run through the deferred functions.
        t2 = time.clock()
        stats['pass2'] = stats.get('pass2', 0.0) + t2-t1

    def pass3(self, node):
        
        global stats
        t1 = time.clock()
        self.pass_n = 3
            # handleNode will raise an exception if it is called.
        self.runDeferred(self._deferredAssignments)
            # Run all queued calls to checkUnusedAssignments and
            # (If Python 2) checkReturnWithArgumentInsideGenerator
        self._deferredAssignments = None
            # Set _deferredAssignments to None so that deferAssignment will fail
            # noisily if called after we've run through the deferred assignments.
        t2 = time.clock()
        stats['pass3'] = stats.get('pass3', 0.0) + t2-t1

    def checkDeadScopes(self):
        """
        Look at scopes which have been fully examined and report names in them
        which were imported but unused.
        """
        global stats
        t1 = time.clock()
        self.pass_n = 4
            # This will raise an exception in handleNode if any nodes are visited.
        for scope in self.deadScopes:
            if isinstance(scope.get('__all__'), ExportBinding):
                all_names = set(scope['__all__'].names)
                if not scope.importStarred and \
                   os.path.basename(self.filename) != '__init__.py':
                    # Look for possible mistakes in the export list
                    undefined = all_names.difference(scope)
                    for name in undefined:
                        self.report(messages.UndefinedExport,
                                    scope['__all__'].source, name)
            else:
                all_names = []
            # Look for imported names that aren't used.
            for value in scope.values():
                if isinstance(value, Importation):
                    used = value.used or value.name in all_names
                    if not used:
                        messg = messages.UnusedImport
                        self.report(messg, value.source, value.name)
                    for node in value.redefined:
                        if isinstance(self.getParent(node), ast.For):
                            messg = messages.ImportShadowedByLoopVar
                        elif used:
                            continue
                        else:
                            messg = messages.RedefinedWhileUnused
                        self.report(messg, node, value.name, value.source)
        t2 = time.clock()
        stats['pass4'] = stats.get('pass4', 0.0) + t2-t1

    def NAME(self, node):
        """
        Handle occurrence of Name (which can be a load/store/delete access.)
        """
        # Locate the name in locals / function / globals scopes.
        if isinstance(node.ctx, (ast.Load, ast.AugLoad)):
            self.handleNodeLoad(node)
            if (node.id == 'locals' and
                isinstance(self.scope, FunctionScope) and
                isinstance(node.parent, ast.Call)
            ):
                # we are doing locals() call in current scope
                self.scope.usesLocals = True
                    # EKR: why does this matter???
        elif isinstance(node.ctx, (ast.Store, ast.AugStore)):
            self.handleNodeStore(node)
        elif isinstance(node.ctx, ast.Del):
            self.handleNodeDelete(node)
        else:
            # must be a Param context -- this only happens for names in function
            # arguments, but these aren't dispatched through here
            raise RuntimeError("Got impossible expression context: %r" % (node.ctx,))
            
    if aft:
        Name = NAME


    # EKR: ctx is Del.
    def handleNodeDelete(self, node):

        def on_conditional_branch():
            """
            Return `True` if node is part of a conditional body.
            """
            current = getattr(node, 'parent', None)
            while current:
                if isinstance(current, (ast.If, ast.While, ast.IfExp)):
                    return True
                current = getattr(current, 'parent', None)
            return False

        name = getNodeName(node)
        if not name:
            return

        if on_conditional_branch():
            # We can not predict if this conditional branch is going to
            # be executed.
            return

        if isinstance(self.scope, FunctionScope) and name in self.scope.globals:
            self.scope.globals.remove(name)
        else:
            try:
                del self.scope[name]
            except KeyError:
                self.report(messages.UndefinedName, node, name)

    def handleNodeLoad(self, node):
        
        global n_load, test_scope
        trace = False # and test_scope == 'test'
        # EKR: ctx is Load or AugLoad.
        name = getNodeName(node)
        if not name:
            return
        # try local scope
        try:
            self.scope[name].used = (self.scope, node)
        except KeyError:
            pass
        else:
            # EKR: the name is in the scope,
            # scope[name] is a Binding, and we have just marked it used.
            if trace: g.trace('pass: %s %40s in %s' % (
                # self.pass_n, name, repr(self.scope)))
                self.pass_n, self.scope[name], repr(self.scope)))
            return

        # EKR: Create a list of previous defining scopes.
        n_load += 1
        defining_scopes = (FunctionScope, ModuleScope, GeneratorScope) # EKR
        scopes = [scope for scope in self.scopeStack[:-1]
            if isinstance(scope, defining_scopes)]
                
        if isinstance(self.scope, GeneratorScope) and scopes[-1] != self.scopeStack[-2]:
            scopes.append(self.scopeStack[-2])

        # try enclosing function scopes and global scope
        importStarred = self.scope.importStarred
        for scope in reversed(scopes):
            importStarred = importStarred or scope.importStarred
            try:
                scope[name].used = (self.scope, node)
            except KeyError:
                pass
            else:
                if trace: g.trace('pass: %s %40s in %s' % (
                self.pass_n, scope[name], repr(scope)))
                return

        # look in the built-ins
        if importStarred or name in self.builtIns:
            return
        if name == '__path__' and os.path.basename(self.filename) == '__init__.py':
            # the special name __path__ is valid only in packages
            return

        # protected with a NameError handler?
        if 'NameError' not in self.exceptHandlers[-1]:
            self.report(messages.UndefinedName, node, name)

    # EKR: called by Name and ExceptHandler.
    # EKR: ctx is Store or AugStore.

    def handleNodeStore(self, node):
        
        global n_store, test_scope
        trace = False # and test_scope == 'test'
        name = getNodeName(node)
        if not name:
            return
        # if the name hasn't already been defined in the current scope
        if isinstance(self.scope, FunctionScope) and name not in self.scope:
            # for each function or module scope above us
            n_store += 1
            for scope in self.scopeStack[:-1]:
                if not isinstance(scope, (FunctionScope, ModuleScope)):
                    continue
                # if the name was defined in that scope, and the name has
                # been accessed already in the current scope, and hasn't
                # been declared global
                used = name in scope and scope[name].used
                if trace: g.trace(name, 'used:', used, scope.name,
                    'isGlobal', name  in self.scope.globals)
                if used and used[0] is self.scope and name not in self.scope.globals:
                    # then it's probably a mistake
                    self.report(messages.UndefinedLocal,
                                scope[name].used[1], name, scope[name].source)
                    break

        parent_stmt = self.getParent(node)
        if (isinstance(parent_stmt, (ast.For, ast.comprehension)) or
            (parent_stmt != node.parent and not self.isLiteralTupleUnpacking(parent_stmt))
        ):
            binding = Binding(name, node)
        elif name == '__all__' and isinstance(self.scope, ModuleScope):
            binding = ExportBinding(name, node.parent, self.scope)
        else:
            binding = Assignment(name, node)
        self.addBinding(node, binding)

    def isLiteralTupleUnpacking(self, node):
        if isinstance(node, ast.Assign):
            for child in node.targets + [node.value]:
                if not hasattr(child, 'elts'):
                    return False
            return True

    def RETURN(self, node):
        
        # if isinstance(self.scope, ClassScope):
        if isinstance(self.scope, (ClassScope, ModuleScope)):
            # EKR: A real bug fix.
            self.report(messages.ReturnOutsideFunction, node)
            return
        if (
            node.value and
            hasattr(self.scope, 'returnValue') and
            not self.scope.returnValue
        ):
            self.scope.returnValue = node.value
        if node.value: # EKR
            self.handleNode(node.value, node)

    if aft:
        Return = RETURN

    def TRY(self, node):
        handler_names = []
        # List the exception handlers
        for handler in node.handlers:
            if isinstance(handler.type, ast.Tuple):
                for exc_type in handler.type.elts:
                    handler_names.append(getNodeName(exc_type))
            elif handler.type:
                handler_names.append(getNodeName(handler.type))
        # Memorize the except handlers and process the body
        self.exceptHandlers.append(handler_names)
        for child in node.body:
            self.handleNode(child, node)
        self.exceptHandlers.pop()
        # Process the other nodes: "except:", "else:", "finally:"
        if aft:
            for field in ('handlers', 'orelse', 'finalbody'):
                for z in getattr(node, field, []):
                    self.handleNode(z, node)
        else:
            self.handleChildren(node, omit='body')
        
    if aft:
        Try = TryExcept = TRY

    TRYEXCEPT = TRY

    def YIELD(self, node):
        self.scope.isGenerator = True
        self.handleNode(node.value, node)
        
    if aft:
        Yield = Await = YieldFrom = YIELD

    AWAIT = YIELDFROM = YIELD
