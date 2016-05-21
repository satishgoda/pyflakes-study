'''AST (Abstract Syntax Tree) related classes.'''
import ast
import xml.sax.saxutils as saxutils
import textwrap
import token as token_module
import leo.core.leoGlobals as g
class AstDumper(object):
    '''
    Return a formatted dump (a string) of the AST node.

    Adapted from Python's ast.dump.

    annotate_fields:    True: show names of fields (can't eval the dump).
    disabled_field:     List of names of fields not to show: e.g. ['ctx',]
    include_attributes: True: show line numbers and column offsets.
    indent:             Number of spaces for each indent.
    '''
    def __init__(self, u, annotate_fields, disabled_fields, format, include_attributes, indent_ws):
        '''Ctor for AstDumper class.'''
        self.u = u
        self.annotate_fields = annotate_fields
        self.disabled_fields = disabled_fields
        self.format = format
        self.include_attributes = include_attributes
        self.indent_ws = indent_ws
    def dump(self, node, level=0):
        sep1 = '\n%s' % (self.indent_ws * (level + 1))
        if isinstance(node, ast.AST):
            fields = [(a, self.dump(b, level + 1)) for a, b in self.get_fields(node)]
                # ast.iter_fields(node)]
            if self.include_attributes and node._attributes:
                fields.extend([(a, self.dump(getattr(node, a), level + 1))
                    for a in node._attributes])
            # Not used at present.
            # aList = self.extra_attributes(node)
            # if aList: fields.extend(aList)
            if self.annotate_fields:
                aList = ['%s=%s' % (a, b) for a, b in fields]
            else:
                aList = [b for a, b in fields]
            compressed = not any([isinstance(b, list) and len(b) > 1 for a, b in fields])
            name = node.__class__.__name__
            if compressed and len(','.join(aList)) < 100:
                return '%s(%s)' % (name, ','.join(aList))
            else:
                sep = '' if len(aList) <= 1 else sep1
                return '%s(%s%s)' % (name, sep, sep1.join(aList))
        elif isinstance(node, list):
            compressed = not any([isinstance(z, list) and len(z) > 1 for z in node])
            sep = '' if compressed and len(node) <= 1 else sep1
            return '[%s]' % ''.join(
                ['%s%s' % (sep, self.dump(z, level + 1)) for z in node])
        else:
            return repr(node)
    def get_fields(self, node):
        fields = [z for z in ast.iter_fields(node)]
        result = []
        for a, b in fields:
            if a not in self.disabled_fields:
                if b not in (None, []):
                    result.append((a, b),)
        return result
    def extra_attributes(self, node):
        '''Return the tuple (field,repr(field)) for all extra fields.'''
        d = {
            # 'e': self.do_repr,
            # 'cache':self.do_cache_list,
            # 'reach':self.do_reaching_list,
            # 'typ':  self.do_types_list,
        }
        aList = []
        for attr in sorted(d.keys()):
            if hasattr(node, attr):
                val = getattr(node, attr)
                f = d.get(attr)
                s = f(attr, node, val)
                if s:
                    aList.append((attr, s),)
        return aList
    def do_cache_list(self, attr, node, val):
        return self.u.dump_cache(node)
    def do_reaching_list(self, attr, node, val):
        assert attr == 'reach'
        return '[%s]' % ','.join(
            [self.format(z).strip() or repr(z)
                for z in getattr(node, attr)])
    def do_repr(self, attr, node, val):
        return repr(val)
    def do_types_list(self, attr, node, val):
        assert attr == 'typ'
        return '[%s]' % ','.join(
            [repr(z) for z in getattr(node, attr)])
class AstFormatter(object):
    '''
    A class to recreate source code from an AST.

    This does not have to be perfect, but it should be close.

    Also supports optional annotations such as line numbers, file names, etc.
    '''
    # No ctor.
    # pylint: disable=consider-using-enumerate
    # def __call__(self,node):
        # '''__call__ method for AstFormatter class.'''
        # return self.format(node)
    def format(self, node):
        '''Format the node (or list of nodes) and its descendants.'''
        self.level = 0
        val = self.visit(node)
        return val and val.strip() or ''
    def visit(self, node):
        '''Return the formatted version of an Ast node, or list of Ast nodes.'''
        if isinstance(node, (list, tuple)):
            return ','.join([self.visit(z) for z in node])
        elif node is None:
            return 'None'
        else:
            assert isinstance(node, ast.AST), node.__class__.__name__
            method_name = 'do_' + node.__class__.__name__
            method = getattr(self, method_name)
            s = method(node)
            assert g.isString(s), type(s)
            return s
    # 2: ClassDef(identifier name, expr* bases,
    #             stmt* body, expr* decorator_list)
    # 3: ClassDef(identifier name, expr* bases,
    #             keyword* keywords, expr? starargs, expr? kwargs
    #             stmt* body, expr* decorator_list)
    #
    # keyword arguments supplied to call (NULL identifier for **kwargs)
    # keyword = (identifier? arg, expr value)

    def do_ClassDef(self, node):

        result = []
        name = node.name # Only a plain string is valid.
        bases = [self.visit(z) for z in node.bases] if node.bases else []
        if getattr(node, 'keywords', None): # Python 3
            for keyword in node.keywords:
                bases.append('%s=%s' % (keyword.arg, self.visit(keyword.value)))
        if getattr(node, 'starargs', None): # Python 3
            bases.append('*%s', self.visit(node.starargs))
        if getattr(node, 'kwargs', None): # Python 3
            bases.append('*%s', self.visit(node.kwargs))
        if bases:
            result.append(self.indent('class %s(%s):\n' % (name, ','.join(bases))))
        else:
            result.append(self.indent('class %s:\n' % name))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        return ''.join(result)
    # 2: FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list)
    # 3: FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list,
    #                expr? returns)

    def do_FunctionDef(self, node):
        '''Format a FunctionDef node.'''
        result = []
        if node.decorator_list:
            for z in node.decorator_list:
                result.append('@%s\n' % self.visit(z))
        name = node.name # Only a plain string is valid.
        args = self.visit(node.args) if node.args else ''
        if getattr(node, 'returns', None): # Python 3.
            returns = self.visit(node.returns)
            result.append(self.indent('def %s(%s): -> %s\n' % (name, args, returns)))
        else:
            result.append(self.indent('def %s(%s):\n' % (name, args)))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        return ''.join(result)
    def do_Interactive(self, node):
        for z in node.body:
            self.visit(z)
    def do_Module(self, node):
        assert 'body' in node._fields
        result = ''.join([self.visit(z) for z in node.body])
        return result # 'module:\n%s' % (result)
    def do_Lambda(self, node):
        return self.indent('lambda %s: %s' % (
            self.visit(node.args),
            self.visit(node.body)))
    def do_Expr(self, node):
        '''An outer expression: must be indented.'''
        return self.indent('%s\n' % self.visit(node.value))
    def do_Expression(self, node):
        '''An inner expression: do not indent.'''
        return '%s\n' % self.visit(node.body)
    def do_GeneratorExp(self, node):
        elt = self.visit(node.elt) or ''
        gens = [self.visit(z) for z in node.generators]
        gens = [z if z else '<**None**>' for z in gens] # Kludge: probable bug.
        return '<gen %s for %s>' % (elt, ','.join(gens))
    def do_AugLoad(self, node):
        return 'AugLoad'

    def do_Del(self, node):
        return 'Del'

    def do_Load(self, node):
        return 'Load'

    def do_Param(self, node):
        return 'Param'

    def do_Store(self, node):
        return 'Store'
    # 2: arguments = (expr* args, identifier? vararg, identifier?
    #                arg? kwarg, expr* defaults)
    # 3: arguments = (arg*  args, arg? vararg,
    #                arg* kwonlyargs, expr* kw_defaults,
    #                arg? kwarg, expr* defaults)

    def do_arguments(self, node):
        '''Format the arguments node.'''
        kind = self.kind(node)
        assert kind == 'arguments', kind
        args = [self.visit(z) for z in node.args]
        defaults = [self.visit(z) for z in node.defaults]
        args2 = []
        n_plain = len(args) - len(defaults)
        for i in range(len(args)):
            if i < n_plain:
                args2.append(args[i])
            else:
                args2.append('%s=%s' % (args[i], defaults[i - n_plain]))
        if g.isPython3:
            args  = [self.visit(z) for z in node.kwonlyargs]
            defaults = [self.visit(z) for z in node.kw_defaults]
            n_plain = len(args) - len(defaults)
            for i in range(len(args)):
                if i < n_plain:
                    args2.append(args[i])
                else:
                    args2.append('%s=%s' % (args[i], defaults[i - n_plain]))
            # Add the vararg and kwarg expressions.
            vararg = getattr(node, 'vararg', None)
            if vararg: args2.append('*' + self.visit(vararg))
            kwarg = getattr(node, 'kwarg', None)
            if kwarg: args2.append('**' + self.visit(kwarg))
        else:
            # Add the vararg and kwarg names.
            name = getattr(node, 'vararg', None)
            if name: args2.append('*' + name)
            name = getattr(node, 'kwarg', None)
            if name: args2.append('**' + name)
        return ','.join(args2)
    # 3: arg = (identifier arg, expr? annotation)

    def do_arg(self, node):
        if getattr(node, 'annotation', None):
            return self.visit(node.annotation)
        else:
            return ''
    # Attribute(expr value, identifier attr, expr_context ctx)

    def do_Attribute(self, node):
        return '%s.%s' % (
            self.visit(node.value),
            node.attr) # Don't visit node.attr: it is always a string.
    def do_Bytes(self, node): # Python 3.x only.
        assert g.isPython3
        return str(node.s)
    # Call(expr func, expr* args, keyword* keywords, expr? starargs, expr? kwargs)

    def do_Call(self, node):
        # g.trace(node,Utils().dump_ast(node))
        func = self.visit(node.func)
        args = [self.visit(z) for z in node.args]
        for z in node.keywords:
            # Calls f.do_keyword.
            args.append(self.visit(z))
        if getattr(node, 'starargs', None):
            args.append('*%s' % (self.visit(node.starargs)))
        if getattr(node, 'kwargs', None):
            args.append('**%s' % (self.visit(node.kwargs)))
        args = [z for z in args if z] # Kludge: Defensive coding.
        return '%s(%s)' % (func, ','.join(args))
    # keyword = (identifier arg, expr value)

    def do_keyword(self, node):
        # node.arg is a string.
        value = self.visit(node.value)
        # This is a keyword *arg*, not a Python keyword!
        return '%s=%s' % (node.arg, value)
    def do_comprehension(self, node):
        result = []
        name = self.visit(node.target) # A name.
        it = self.visit(node.iter) # An attribute.
        result.append('%s in %s' % (name, it))
        ifs = [self.visit(z) for z in node.ifs]
        if ifs:
            result.append(' if %s' % (''.join(ifs)))
        return ''.join(result)
    def do_Dict(self, node):
        result = []
        keys = [self.visit(z) for z in node.keys]
        values = [self.visit(z) for z in node.values]
        if len(keys) == len(values):
            result.append('{\n' if keys else '{')
            items = []
            for i in range(len(keys)):
                items.append('  %s:%s' % (keys[i], values[i]))
            result.append(',\n'.join(items))
            result.append('\n}' if keys else '}')
        else:
            print('Error: f.Dict: len(keys) != len(values)\nkeys: %s\nvals: %s' % (
                repr(keys), repr(values)))
        return ''.join(result)
    def do_Ellipsis(self, node):
        return '...'
    def do_ExtSlice(self, node):
        return ':'.join([self.visit(z) for z in node.dims])
    def do_Index(self, node):
        return self.visit(node.value)
    def do_List(self, node):
        # Not used: list context.
        # self.visit(node.ctx)
        elts = [self.visit(z) for z in node.elts]
        elts = [z for z in elts if z] # Defensive.
        return '[%s]' % ','.join(elts)
    def do_ListComp(self, node):
        elt = self.visit(node.elt)
        gens = [self.visit(z) for z in node.generators]
        gens = [z if z else '<**None**>' for z in gens] # Kludge: probable bug.
        return '%s for %s' % (elt, ''.join(gens))
    def do_Name(self, node):
        return node.id

    def do_NameConstant(self, node): # Python 3 only.
        s = repr(node.value)
        return 'bool' if s in ('True', 'False') else s
    def do_Num(self, node):
        return repr(node.n)
    # Python 2.x only

    def do_Repr(self, node):
        return 'repr(%s)' % self.visit(node.value)
    def do_Slice(self, node):
        lower, upper, step = '', '', ''
        if getattr(node, 'lower', None) is not None:
            lower = self.visit(node.lower)
        if getattr(node, 'upper', None) is not None:
            upper = self.visit(node.upper)
        if getattr(node, 'step', None) is not None:
            step = self.visit(node.step)
        if step:
            return '%s:%s:%s' % (lower, upper, step)
        else:
            return '%s:%s' % (lower, upper)
    def do_Str(self, node):
        '''This represents a string constant.'''
        return repr(node.s)
    # Subscript(expr value, slice slice, expr_context ctx)

    def do_Subscript(self, node):
        value = self.visit(node.value)
        the_slice = self.visit(node.slice)
        return '%s[%s]' % (value, the_slice)
    def do_Tuple(self, node):
        elts = [self.visit(z) for z in node.elts]
        return '(%s)' % ','.join(elts)
    def do_BinOp(self, node):
        return '%s%s%s' % (
            self.visit(node.left),
            self.op_name(node.op),
            self.visit(node.right))
    def do_BoolOp(self, node):
        op_name = self.op_name(node.op)
        values = [self.visit(z) for z in node.values]
        return op_name.join(values)
    def do_Compare(self, node):
        result = []
        lt = self.visit(node.left)
        # ops   = [self.visit(z) for z in node.ops]
        ops = [self.op_name(z) for z in node.ops]
        comps = [self.visit(z) for z in node.comparators]
        result.append(lt)
        if len(ops) == len(comps):
            for i in range(len(ops)):
                result.append('%s%s' % (ops[i], comps[i]))
        else:
            g.trace('ops', repr(ops), 'comparators', repr(comps))
        return ''.join(result)
    def do_UnaryOp(self, node):
        return '%s%s' % (
            self.op_name(node.op),
            self.visit(node.operand))
    def do_IfExp(self, node):
        return '%s if %s else %s ' % (
            self.visit(node.body),
            self.visit(node.test),
            self.visit(node.orelse))
    def do_Assert(self, node):
        test = self.visit(node.test)
        if getattr(node, 'msg', None):
            message = self.visit(node.msg)
            return self.indent('assert %s, %s' % (test, message))
        else:
            return self.indent('assert %s' % test)
    def do_Assign(self, node):
        return self.indent('%s=%s\n' % (
            '='.join([self.visit(z) for z in node.targets]),
            self.visit(node.value)))
    def do_AugAssign(self, node):
        return self.indent('%s%s=%s\n' % (
            self.visit(node.target),
            self.op_name(node.op), # Bug fix: 2013/03/08.
            self.visit(node.value)))
    def do_Break(self, node):
        return self.indent('break\n')
    def do_Continue(self, node):
        return self.indent('continue\n')
    def do_Delete(self, node):
        targets = [self.visit(z) for z in node.targets]
        return self.indent('del %s\n' % ','.join(targets))
    def do_ExceptHandler(self, node):
        result = []
        result.append(self.indent('except'))
        if getattr(node, 'type', None):
            result.append(' %s' % self.visit(node.type))
        if getattr(node, 'name', None):
            if isinstance(node.name, ast.AST):
                result.append(' as %s' % self.visit(node.name))
            else:
                result.append(' as %s' % node.name) # Python 3.x.
        result.append(':\n')
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        return ''.join(result)
    # Python 2.x only

    def do_Exec(self, node):
        body = self.visit(node.body)
        args = [] # Globals before locals.
        if getattr(node, 'globals', None):
            args.append(self.visit(node.globals))
        if getattr(node, 'locals', None):
            args.append(self.visit(node.locals))
        if args:
            return self.indent('exec %s in %s\n' % (
                body, ','.join(args)))
        else:
            return self.indent('exec %s\n' % (body))
    def do_For(self, node):
        result = []
        result.append(self.indent('for %s in %s:\n' % (
            self.visit(node.target),
            self.visit(node.iter))))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        if node.orelse:
            result.append(self.indent('else:\n'))
            for z in node.orelse:
                self.level += 1
                result.append(self.visit(z))
                self.level -= 1
        return ''.join(result)
    def do_Global(self, node):
        return self.indent('global %s\n' % (
            ','.join(node.names)))
    def do_If(self, node):
        result = []
        result.append(self.indent('if %s:\n' % (
            self.visit(node.test))))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        if node.orelse:
            result.append(self.indent('else:\n'))
            for z in node.orelse:
                self.level += 1
                result.append(self.visit(z))
                self.level -= 1
        return ''.join(result)
    def do_Import(self, node):
        names = []
        for fn, asname in self.get_import_names(node):
            if asname:
                names.append('%s as %s' % (fn, asname))
            else:
                names.append(fn)
        return self.indent('import %s\n' % (
            ','.join(names)))
    def get_import_names(self, node):
        '''Return a list of the the full file names in the import statement.'''
        result = []
        for ast2 in node.names:
            if self.kind(ast2) == 'alias':
                data = ast2.name, ast2.asname
                result.append(data)
            else:
                g.trace('unsupported kind in Import.names list', self.kind(ast2))
        return result
    def do_ImportFrom(self, node):
        names = []
        for fn, asname in self.get_import_names(node):
            if asname:
                names.append('%s as %s' % (fn, asname))
            else:
                names.append(fn)
        return self.indent('from %s import %s\n' % (
            node.module,
            ','.join(names)))
    # Nonlocal(identifier* names)

    def do_Nonlocal(self, node):

        return self.indent('nonlocal %s\n' % ', '.join(node.names))
    def do_Pass(self, node):
        return self.indent('pass\n')
    # Python 2.x only

    def do_Print(self, node):
        vals = []
        for z in node.values:
            vals.append(self.visit(z))
        if getattr(node, 'dest', None):
            vals.append('dest=%s' % self.visit(node.dest))
        if getattr(node, 'nl', None):
            # vals.append('nl=%s' % self.visit(node.nl))
            vals.append('nl=%s' % node.nl)
        return self.indent('print(%s)\n' % (
            ','.join(vals)))
    def do_Raise(self, node):
        args = []
        for attr in ('type', 'inst', 'tback'):
            if getattr(node, attr, None) is not None:
                args.append(self.visit(getattr(node, attr)))
        if args:
            return self.indent('raise %s\n' % (
                ','.join(args)))
        else:
            return self.indent('raise\n')
    def do_Return(self, node):
        if node.value:
            return self.indent('return %s\n' % (
                self.visit(node.value)))
        else:
            return self.indent('return\n')
    # Starred(expr value, expr_context ctx)

    def do_Starred(self, node):

        return '*' + self.visit(node.value)
    # def do_Suite(self,node):
        # for z in node.body:
            # s = self.visit(z)
    # Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)

    def do_Try(self, node): # Python 3

        result = []
        self.append(self.indent('try:\n'))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        if node.handlers:
            for z in node.handlers:
                result.append(self.visit(z))
        if node.orelse:
            result.append(self.indent('else:\n'))
            for z in node.orelse:
                self.level += 1
                result.append(self.visit(z))
                self.level -= 1
        if node.finalbody:
            result.append(self.indent('finally:\n'))
            for z in node.finalbody:
                self.level += 1
                result.append(self.visit(z))
                self.level -= 1
        return ''.join(result)
    def do_TryExcept(self, node):
        result = []
        result.append(self.indent('try:\n'))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        if node.handlers:
            for z in node.handlers:
                result.append(self.visit(z))
        if node.orelse:
            result.append('else:\n')
            for z in node.orelse:
                self.level += 1
                result.append(self.visit(z))
                self.level -= 1
        return ''.join(result)
    def do_TryFinally(self, node):
        result = []
        result.append(self.indent('try:\n'))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        result.append(self.indent('finally:\n'))
        for z in node.finalbody:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        return ''.join(result)
    def do_While(self, node):
        result = []
        result.append(self.indent('while %s:\n' % (
            self.visit(node.test))))
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        if node.orelse:
            result.append('else:\n')
            for z in node.orelse:
                self.level += 1
                result.append(self.visit(z))
                self.level -= 1
        return ''.join(result)
    # 2:  With(expr context_expr, expr? optional_vars,
    #          stmt* body)
    # 3:  With(withitem* items,
    #          stmt* body)
    # withitem = (expr context_expr, expr? optional_vars)

    def do_With(self, node):
        result = []
        result.append(self.indent('with '))
        if getattr(node, 'context_expression', None):
            result.append(self.visit(node.context_expresssion))
        vars_list = []
        if getattr(node, 'optional_vars', None):
            try:
                for z in node.optional_vars:
                    vars_list.append(self.visit(z))
            except TypeError: # Not iterable.
                vars_list.append(self.visit(node.optional_vars))
        if getattr(node, 'items', None): # Python 3.
            for item in node.items:
                result.append(self.visit(item.context_expr))
                if getattr(item, 'optional_vars', None):
                    try:
                        for z in item.optional_vars:
                            vars_list.append(self.visit(z))
                    except TypeError: # Not iterable.
                        vars_list.append(self.visit(item.optional_vars))
        result.append(','.join(vars_list))
        result.append(':\n')
        for z in node.body:
            self.level += 1
            result.append(self.visit(z))
            self.level -= 1
        result.append('\n')
        return ''.join(result)
    def do_Yield(self, node):
        if getattr(node, 'value', None):
            return self.indent('yield %s\n' % (
                self.visit(node.value)))
        else:
            return self.indent('yield\n')
    # YieldFrom(expr value)

    def do_YieldFrom(self, node):

        return self.indent('yield from %s\n' % (
            self.visit(node.value)))
    def kind(self, node):
        '''Return the name of node's class.'''
        return node.__class__.__name__
    def indent(self, s):
        return '%s%s' % (' ' * 4 * self.level, s)

    def op_name (self,node,strict=True):
        '''Return the print name of an operator node.'''
        d = {
            # Binary operators.
            'Add':       '+',
            'BitAnd':    '&',
            'BitOr':     '|',
            'BitXor':    '^',
            'Div':       '/',
            'FloorDiv':  '//',
            'LShift':    '<<',
            'Mod':       '%',
            'Mult':      '*',
            'Pow':       '**',
            'RShift':    '>>',
            'Sub':       '-',
            # Boolean operators.
            'And':   ' and ',
            'Or':    ' or ',
            # Comparison operators
            'Eq':    '==',
            'Gt':    '>',
            'GtE':   '>=',
            'In':    ' in ',
            'Is':    ' is ',
            'IsNot': ' is not ',
            'Lt':    '<',
            'LtE':   '<=',
            'NotEq': '!=',
            'NotIn': ' not in ',
            # Context operators.
            'AugLoad':  '<AugLoad>',
            'AugStore': '<AugStore>',
            'Del':      '<Del>',
            'Load':     '<Load>',
            'Param':    '<Param>',
            'Store':    '<Store>',
            # Unary operators.
            'Invert':   '~',
            'Not':      ' not ',
            'UAdd':     '+',
            'USub':     '-',
        }
        name = d.get(self.kind(node),'<%s>' % node.__class__.__name__)
        if strict: assert name,self.kind(node)
        return name
class AstFullTraverser(object):
    '''
    A fast traverser for AST trees: it visits every node (except node.ctx fields).

    Sets .context and .parent ivars before visiting each node.
    '''

    def __init__(self):
        '''Ctor for AstFullTraverser class.'''
        self.context = None
        self.parent = None
        self.trace = False
    # 2: ClassDef(identifier name, expr* bases, stmt* body, expr* decorator_list)
    # 3: ClassDef(identifier name, expr* bases,
    #             keyword* keywords, expr? starargs, expr? kwargs
    #             stmt* body, expr* decorator_list)
    #
    # keyword arguments supplied to call (NULL identifier for **kwargs)
    # keyword = (identifier? arg, expr value)

    def do_ClassDef(self, node):
        old_context = self.context
        self.context = node
        for z in node.bases:
            self.visit(z)
        if getattr(node, 'keywords', None): # Python 3
            for keyword in node.keywords:
                self.visit(keyword.value)
        if getattr(node, 'starargs', None): # Python 3
            self.visit(node.starargs)
        if getattr(node, 'kwargs', None): # Python 3
            self.visit(node.kwargs)
        for z in node.body:
            self.visit(z)
        for z in node.decorator_list:
            self.visit(z)
        self.context = old_context
    # 2: FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list)
    # 3: FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list,
    #                expr? returns)

    def do_FunctionDef(self, node):

        old_context = self.context
        self.context = node
        # Visit the tree in token order.
        for z in node.decorator_list:
            self.visit(z)
        assert g.isString(node.name)
        self.visit(node.args)
        if getattr(node, 'returns', None): # Python 3.
            self.visit(node.returns)
        for z in node.body:
            self.visit(z)
        self.context = old_context
    def do_Interactive(self, node):
        assert False, 'Interactive context not supported'
    # Lambda(arguments args, expr body)

    def do_Lambda(self, node):
        old_context = self.context
        self.context = node
        self.visit(node.args)
        self.visit(node.body)
        self.context = old_context
    def do_Module(self, node):
        self.context = node
        for z in node.body:
            self.visit(z)
        self.context = None
    # Not used in this class, but may be called by subclasses.

    def do_AugLoad(self, node):
        pass

    def do_Del(self, node):
        pass

    def do_Load(self, node):
        pass

    def do_Param(self, node):
        pass

    def do_Store(self, node):
        pass
    def kind(self, node):
        return node.__class__.__name__
    # 2: arguments = (expr* args, identifier? vararg,
    #                 identifier? kwarg, expr* defaults)
    # 3: arguments = (arg*  args, arg? vararg,
    #                 arg* kwonlyargs, expr* kw_defaults,
    #                 arg? kwarg, expr* defaults)

    def do_arguments(self, node):

        for z in node.args:
            self.visit(z)
        if g.isPython3 and getattr(node, 'vararg', None):
            # An identifier in Python 2.
            self.visit(node.vararg)
        if getattr(node, 'kwonlyargs', None): # Python 3.
            self.visit_list(node.kwonlyargs)
        if getattr(node, 'kw_defaults', None): # Python 3.
            self.visit_list(node.kw_defaults)
        if g.isPython3 and getattr(node, 'kwarg', None):
            # An identifier in Python 2.
            self.visit(node.kwarg)
        for z in node.defaults:
            self.visit(z)

    # 3: arg = (identifier arg, expr? annotation)

    def do_arg(self, node):
        if getattr(node, 'annotation', None):
            self.visit(node.annotation)
    # Attribute(expr value, identifier attr, expr_context ctx)

    def do_Attribute(self, node):
        self.visit(node.value)
        # self.visit(node.ctx)
    # BinOp(expr left, operator op, expr right)

    def do_BinOp(self, node):
        self.visit(node.left)
        # self.op_name(node.op)
        self.visit(node.right)
    # BoolOp(boolop op, expr* values)

    def do_BoolOp(self, node):
        for z in node.values:
            self.visit(z)
    def do_Bytes(self, node):
        pass # Python 3.x only.
    # Call(expr func, expr* args, keyword* keywords, expr? starargs, expr? kwargs)

    def do_Call(self, node):
        # Call the nodes in token order.
        self.visit(node.func)
        for z in node.args:
            self.visit(z)
        for z in node.keywords:
            self.visit(z)
        if getattr(node, 'starargs', None):
            self.visit(node.starargs)
        if getattr(node, 'kwargs', None):
            self.visit(node.kwargs)
    # Compare(expr left, cmpop* ops, expr* comparators)

    def do_Compare(self, node):
        # Visit all nodes in token order.
        self.visit(node.left)
        assert len(node.ops) == len(node.comparators)
        for i in range(len(node.ops)):
            self.visit(node.ops[i])
            self.visit(node.comparators[i])
        # self.visit(node.left)
        # for z in node.comparators:
            # self.visit(z)
    # Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

    def do_Eq(self, node): pass

    def do_Gt(self, node): pass

    def do_GtE(self, node): pass

    def do_In(self, node): pass

    def do_Is(self, node): pass

    def do_IsNot(self, node): pass

    def do_Lt(self, node): pass

    def do_LtE(self, node): pass

    def do_NotEq(self, node): pass

    def do_NotIn(self, node): pass
    # comprehension (expr target, expr iter, expr* ifs)

    def do_comprehension(self, node):
        self.visit(node.target) # A name.
        self.visit(node.iter) # An attribute.
        for z in node.ifs:
            self.visit(z)
    # Dict(expr* keys, expr* values)

    def do_Dict(self, node):
        # Visit all nodes in token order.
        assert len(node.keys) == len(node.values)
        for i in range(len(node.keys)):
            self.visit(node.keys[i])
            self.visit(node.values[i])
    def do_Ellipsis(self, node):
        pass
    # Expr(expr value)

    def do_Expr(self, node):
        self.visit(node.value)
    def do_Expression(self, node):
        '''An inner expression'''
        self.visit(node.body)
    def do_ExtSlice(self, node):
        for z in node.dims:
            self.visit(z)
    # GeneratorExp(expr elt, comprehension* generators)

    def do_GeneratorExp(self, node):
        self.visit(node.elt)
        for z in node.generators:
            self.visit(z)
    # IfExp(expr test, expr body, expr orelse)

    def do_IfExp(self, node):
        self.visit(node.body)
        self.visit(node.test)
        self.visit(node.orelse)
    def do_Index(self, node):
        self.visit(node.value)
    # keyword = (identifier arg, expr value)

    def do_keyword(self, node):
        # node.arg is a string.
        self.visit(node.value)
    # List(expr* elts, expr_context ctx)

    def do_List(self, node):
        for z in node.elts:
            self.visit(z)
        # self.visit(node.ctx)
    # ListComp(expr elt, comprehension* generators)

    def do_ListComp(self, node):
        self.visit(node.elt)
        for z in node.generators:
            self.visit(z)
    # Name(identifier id, expr_context ctx)

    def do_Name(self, node):
        # self.visit(node.ctx)
        pass

    def do_NameConstant(self, node): # Python 3 only.
        pass
        # s = repr(node.value)
        # return 'bool' if s in ('True', 'False') else s
    def do_Num(self, node):
        pass # Num(object n) # a number as a PyObject.
    # Python 2.x only
    # Repr(expr value)

    def do_Repr(self, node):
        self.visit(node.value)
    def do_Slice(self, node):
        if getattr(node, 'lower', None):
            self.visit(node.lower)
        if getattr(node, 'upper', None):
            self.visit(node.upper)
        if getattr(node, 'step', None):
            self.visit(node.step)
    def do_Str(self, node):
        pass # represents a string constant.
    # Subscript(expr value, slice slice, expr_context ctx)

    def do_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)
        # self.visit(node.ctx)
    # Tuple(expr* elts, expr_context ctx)

    def do_Tuple(self, node):
        for z in node.elts:
            self.visit(z)
        # self.visit(node.ctx)
    # UnaryOp(unaryop op, expr operand)

    def do_UnaryOp(self, node):
        # self.op_name(node.op)
        self.visit(node.operand)
    # identifier name, identifier? asname)

    def do_alias(self, node):
        # self.visit(node.name)
        # if getattr(node,'asname')
            # self.visit(node.asname)
        pass
    # Assert(expr test, expr? msg)

    def do_Assert(self, node):
        self.visit(node.test)
        if node.msg:
            self.visit(node.msg)
    # Assign(expr* targets, expr value)

    def do_Assign(self, node):
        for z in node.targets:
            self.visit(z)
        self.visit(node.value)
    # AugAssign(expr target, operator op, expr value)

    def do_AugAssign(self, node):
        # g.trace('FT',Utils().format(node),g.callers())
        self.visit(node.target)
        self.visit(node.value)
    def do_Break(self, tree):
        pass
    def do_Continue(self, tree):
        pass
    # Delete(expr* targets)

    def do_Delete(self, node):
        for z in node.targets:
            self.visit(z)
    # Python 2: ExceptHandler(expr? type, expr? name, stmt* body)
    # Python 3: ExceptHandler(expr? type, identifier? name, stmt* body)

    def do_ExceptHandler(self, node):
        if node.type:
            self.visit(node.type)
        if node.name and isinstance(node.name, ast.Name):
            self.visit(node.name)
        for z in node.body:
            self.visit(z)
    # Python 2.x only
    # Exec(expr body, expr? globals, expr? locals)

    def do_Exec(self, node):
        self.visit(node.body)
        if getattr(node, 'globals', None):
            self.visit(node.globals)
        if getattr(node, 'locals', None):
            self.visit(node.locals)
    # For(expr target, expr iter, stmt* body, stmt* orelse)

    def do_For(self, node):
        self.visit(node.target)
        self.visit(node.iter)
        for z in node.body:
            self.visit(z)
        for z in node.orelse:
            self.visit(z)
    # Global(identifier* names)

    def do_Global(self, node):
        pass
    # If(expr test, stmt* body, stmt* orelse)

    def do_If(self, node):
        self.visit(node.test)
        for z in node.body:
            self.visit(z)
        for z in node.orelse:
            self.visit(z)
    # Import(alias* names)

    def do_Import(self, node):
        pass
    # ImportFrom(identifier? module, alias* names, int? level)

    def do_ImportFrom(self, node):
        # for z in node.names:
            # self.visit(z)
        pass
    # Nonlocal(identifier* names)

    def do_Nonlocal(self, node):

        pass
    def do_Pass(self, node):
        pass
    # Python 2.x only
    # Print(expr? dest, expr* values, bool nl)

    def do_Print(self, node):
        if getattr(node, 'dest', None):
            self.visit(node.dest)
        for expr in node.values:
            self.visit(expr)
    # Raise(expr? type, expr? inst, expr? tback)

    def do_Raise(self, node):
        if getattr(node, 'type', None):
            self.visit(node.type)
        if getattr(node, 'inst', None):
            self.visit(node.inst)
        if getattr(node, 'tback', None):
            self.visit(node.tback)
    # Return(expr? value)

    def do_Return(self, node):
        if node.value:
            self.visit(node.value)
    # Starred(expr value, expr_context ctx)

    def do_Starred(self, node):

        self.visit(node.value)
    # Python 3 only: Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)

    def do_Try(self, node):
        for z in node.body:
            self.visit(z)
        for z in node.handlers:
            self.visit(z)
        for z in node.orelse:
            self.visit(z)
        for z in node.finalbody:
            self.visit(z)
    # TryExcept(stmt* body, excepthandler* handlers, stmt* orelse)

    def do_TryExcept(self, node):
        for z in node.body:
            self.visit(z)
        for z in node.handlers:
            self.visit(z)
        for z in node.orelse:
            self.visit(z)
    # TryFinally(stmt* body, stmt* finalbody)

    def do_TryFinally(self, node):
        for z in node.body:
            self.visit(z)
        for z in node.finalbody:
            self.visit(z)
    # While(expr test, stmt* body, stmt* orelse)

    def do_While(self, node):
        self.visit(node.test) # Bug fix: 2013/03/23.
        for z in node.body:
            self.visit(z)
        for z in node.orelse:
            self.visit(z)
    # 2:  With(expr context_expr, expr? optional_vars,
    #          stmt* body)
    # 3:  With(withitem* items,
    #          stmt* body)
    # withitem = (expr context_expr, expr? optional_vars)

    def do_With(self, node):
        if getattr(node, 'context_expr', None):
            self.visit(node.context_expr)
        if getattr(node, 'optional_vars', None):
            self.visit(node.optional_vars)
        if getattr(node, 'items', None): # Python 3.
            for item in node.items:
                self.visit(item.context_expr)
                if getattr(item, 'optional_vars', None):
                    try:
                        for z in item.optional_vars:
                            self.visit(z)
                    except TypeError: # Not iterable.
                        self.visit(item.optional_vars)
        for z in node.body:
            self.visit(z)
    #  Yield(expr? value)

    def do_Yield(self, node):
        if node.value:
            self.visit(node.value)
    # YieldFrom(expr value)

    def do_YieldFrom(self, node):

        self.visit(node.value)
    def visit(self, node):
        '''Visit a *single* ast node.  Visitors are responsible for visiting children!'''
        assert isinstance(node, ast.AST), node.__class__.__name__
        trace = False
        # Visit the children with the new parent.
        old_parent = self.parent
        self.parent = node # Bug fix: 2016/05/18.
        method_name = 'do_' + node.__class__.__name__
        method = getattr(self, method_name)
        if trace: g.trace(method_name)
        val = method(node)
        self.parent = old_parent
        return val

    def visit_children(self, node):
        assert False, 'must visit children explicitly'
    def visit_list(self, aList):
        '''Visit all ast nodes in aList.'''
        assert isinstance(aList, (list, tuple)), repr(aList)
        for z in aList:
            self.visit(z)
        return None
class AstPatternFormatter(AstFormatter):
    '''
    A subclass of AstFormatter that replaces values of constants by Bool,
    Bytes, Int, Name, Num or Str.
    '''
    # No ctor.
    # Return generic markers allow better pattern matches.

    def do_BoolOp(self, node): # Python 2.x only.
        return 'Bool'

    def do_Bytes(self, node): # Python 3.x only.
        return 'Bytes' # return str(node.s)

    def do_Name(self, node):
        return 'Bool' if node.id in ('True', 'False') else node.id
        
    def do_NameConstant(self, node): # Python 3 only.
        s = repr(node.value)
        return 'bool' if s in ('True', 'False') else s

    def do_Num(self, node):
        return 'Num' # return repr(node.n)

    def do_Str(self, node):
        '''This represents a string constant.'''
        return 'Str' # return repr(node.s)
class TokenSync(object):
    '''A class to sync and remember tokens.'''
    # To do: handle comments, line breaks...
    def __init__(self, s, tokens):
        '''Ctor for TokenSync class.'''
        assert isinstance(tokens, list) # Not a generator.
        self.s = s
        self.first_leading_line = None
        self.lines = [z.rstrip() for z in g.splitLines(s)]
        # Order is important from here on...
        self.nl_token = self.make_nl_token()
        self.line_tokens = self.make_line_tokens(tokens)
        self.blank_lines = self.make_blank_lines()
        self.string_tokens = self.make_string_tokens()
        self.ignored_lines = self.make_ignored_lines()
    def make_blank_lines(self):
        '''Return of list of line numbers of blank lines.'''
        result = []
        for i, aList in enumerate(self.line_tokens):
            # if any([self.token_kind(z) == 'nl' for z in aList]):
            if len(aList) == 1 and self.token_kind(aList[0]) == 'nl':
                result.append(i)
        return result
    def make_ignored_lines(self):
        '''
        Return a copy of line_tokens containing ignored lines,
        that is, full-line comments or blank lines.
        These are the lines returned by leading_lines().
        '''
        result = []
        for i, aList in enumerate(self.line_tokens):
            for z in aList:
                if self.is_line_comment(z):
                    result.append(z)
                    break
            else:
                if i in self.blank_lines:
                    result.append(self.nl_token)
                else:
                    result.append(None)
        assert len(result) == len(self.line_tokens)
        for i, aList in enumerate(result):
            if aList:
                self.first_leading_line = i
                break
        else:
            self.first_leading_line = len(result)
        return result
    def make_line_tokens(self, tokens):
        '''
        Return a list of lists of tokens for each list in self.lines.
        The strings in self.lines may end in a backslash, so care is needed.
        '''
        trace = False
        n, result = len(self.lines), []
        for i in range(0, n + 1):
            result.append([])
        for token in tokens:
            t1, t2, t3, t4, t5 = token
            kind = token_module.tok_name[t1].lower()
            srow, scol = t3
            erow, ecol = t4
            line = erow - 1 if kind == 'string' else srow - 1
            result[line].append(token)
            if trace: g.trace('%3s %s' % (line, self.dump_token(token)))
        assert len(self.lines) + 1 == len(result), len(result)
        return result
    def make_nl_token(self):
        '''Return a newline token with '\n' as both val and raw_val.'''
        t1 = token_module.NEWLINE
        t2 = '\n'
        t3 = (0, 0) # Not used.
        t4 = (0, 0) # Not used.
        t5 = '\n'
        return t1, t2, t3, t4, t5
    def make_string_tokens(self):
        '''Return a copy of line_tokens containing only string tokens.'''
        result = []
        for aList in self.line_tokens:
            result.append([z for z in aList if self.token_kind(z) == 'string'])
        assert len(result) == len(self.line_tokens)
        return result
    def check_strings(self):
        '''Check that all strings have been consumed.'''
        # g.trace(len(self.string_tokens))
        for i, aList in enumerate(self.string_tokens):
            if aList:
                g.trace('warning: line %s. unused strings: %s' % (i, aList))
    def dump_token(self, token, verbose=False):
        '''Dump the token. It is either a string or a 5-tuple.'''
        if g.isString(token):
            return token
        else:
            t1, t2, t3, t4, t5 = token
            kind = g.toUnicode(token_module.tok_name[t1].lower())
            # raw_val = g.toUnicode(t5)
            val = g.toUnicode(t2)
            if verbose:
                return 'token: %10s %r' % (kind, val)
            else:
                return val
    def is_line_comment(self, token):
        '''Return True if the token represents a full-line comment.'''
        t1, t2, t3, t4, t5 = token
        kind = token_module.tok_name[t1].lower()
        raw_val = t5
        return kind == 'comment' and raw_val.lstrip().startswith('#')
    def join(self, aList, sep=','):
        '''return the items of the list joined by sep string.'''
        tokens = []
        for i, token in enumerate(aList or []):
            tokens.append(token)
            if i < len(aList) - 1:
                tokens.append(sep)
        return tokens
    def last_node(self, node):
        '''Return the node of node's tree with the largest lineno field.'''

        class LineWalker(ast.NodeVisitor):

            def __init__(self):
                '''Ctor for LineWalker class.'''
                self.node = None
                self.lineno = -1

            def visit(self, node):
                '''LineWalker.visit.'''
                if hasattr(node, 'lineno'):
                    if node.lineno > self.lineno:
                        self.lineno = node.lineno
                        self.node = node
                if isinstance(node, list):
                    for z in node:
                        self.visit(z)
                else:
                    self.generic_visit(node)

        w = LineWalker()
        w.visit(node)
        return w.node
    def leading_lines(self, node):
        '''Return a list of the preceding comment and blank lines'''
        # This can be called on arbitrary nodes.
        trace = False
        leading = []
        if hasattr(node, 'lineno'):
            i, n = self.first_leading_line, node.lineno
            while i < n:
                token = self.ignored_lines[i]
                if token:
                    s = self.token_raw_val(token).rstrip() + '\n'
                    leading.append(s)
                    if trace: g.trace('%11s: %s' % (i, s.rstrip()))
                i += 1
            self.first_leading_line = i
        return leading
    def leading_string(self, node):
        '''Return a string containing all lines preceding node.'''
        return ''.join(self.leading_lines(node))
    def line_at(self, node, continued_lines=True):
        '''Return the lines at the node, possibly including continuation lines.'''
        n = getattr(node, 'lineno', None)
        if n is None:
            return '<no line> for %s' % node.__class__.__name__
        elif continued_lines:
            aList, n = [], n - 1
            while n < len(self.lines):
                s = self.lines[n]
                if s.endswith('\\'):
                    aList.append(s[: -1])
                    n += 1
                else:
                    aList.append(s)
                    break
            return ''.join(aList)
        else:
            return self.lines[n - 1]
    def sync_string(self, node):
        '''Return the spelling of the string at the given node.'''
        # g.trace('%-10s %2s: %s' % (' ', node.lineno, self.line_at(node)))
        n = node.lineno
        tokens = self.string_tokens[n - 1]
        if tokens:
            token = tokens.pop(0)
            self.string_tokens[n - 1] = tokens
            return self.token_val(token)
        else:
            g.trace('===== underflow', n, node.s)
            return node.s
    def token_kind(self, token):
        '''Return the token's type.'''
        t1, t2, t3, t4, t5 = token
        return g.toUnicode(token_module.tok_name[t1].lower())

    def token_raw_val(self, token):
        '''Return the value of the token.'''
        t1, t2, t3, t4, t5 = token
        return g.toUnicode(t5)

    def token_val(self, token):
        '''Return the raw value of the token.'''
        t1, t2, t3, t4, t5 = token
        return g.toUnicode(t2)
    def tokens_for_statement(self, node):
        assert isinstance(node, ast.AST), node
        name = node.__class__.__name__
        if hasattr(node, 'lineno'):
            tokens = self.line_tokens[node.lineno - 1]
            g.trace(' '.join([self.dump_token(z) for z in tokens]))
        else:
            g.trace('no lineno', name)
    def trailing_comment(self, node):
        '''
        Return a string containing the trailing comment for the node, if any.
        The string always ends with a newline.
        '''
        if hasattr(node, 'lineno'):
            return self.trailing_comment_at_lineno(node.lineno)
        else:
            g.trace('no lineno', node.__class__.__name__, g.callers())
            return '\n'
    def trailing_comment_at_lineno(self, lineno):
        '''Return any trailing comment at the given node.lineno.'''
        trace = False
        tokens = self.line_tokens[lineno - 1]
        for token in tokens:
            if self.token_kind(token) == 'comment':
                raw_val = self.token_raw_val(token).rstrip()
                if not raw_val.strip().startswith('#'):
                    val = self.token_val(token).rstrip()
                    s = ' %s\n' % val
                    if trace: g.trace(lineno, s.rstrip(), g.callers())
                    return s
        return '\n'
    def trailing_lines(self):
        '''return any remaining ignored lines.'''
        trace = False
        trailing = []
        i = self.first_leading_line
        while i < len(self.ignored_lines):
            token = self.ignored_lines[i]
            if token:
                s = self.token_raw_val(token).rstrip() + '\n'
                trailing.append(s)
                if trace: g.trace('%11s: %s' % (i, s.rstrip()))
            i += 1
        self.first_leading_line = i
        return trailing
