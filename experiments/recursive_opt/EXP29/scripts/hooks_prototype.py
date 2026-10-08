"""Prototype of the recursive_opt hook materializer (shared by options a, b, c)."""
import ast, copy, functools, inspect, pickle

DENY_CALLS = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'setattr', 'delattr'}
DENY_IMPORTS = {'os', 'sys', 'subprocess', 'socket', 'shutil', 'importlib', 'ctypes', 'pathlib'}
NEVER = {'train', 'step', '__init__', 'save', 'load', 'update'}          # used by option (b)/(c)

def hook_points(cls, policy='declared'):
    """Which methods of cls may be replaced."""
    if policy == 'declared':        # (a): union of `hookable_methods` along the MRO
        return {n for k in cls.__mro__ for n in vars(k).get('hookable_methods', ())}
    # (b) 'any': every public or private function of the class, minus a safety denylist
    return {n for n, m in inspect.getmembers(cls, inspect.isfunction) if n not in NEVER and not n.startswith('__')}

def _compile(name, source, original):
    tree = ast.parse(source)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or fns[0].name != name:
        raise ValueError(f'hook {name!r} must be exactly one function named {name}')
    want = list(inspect.signature(original).parameters)
    got = [a.arg for a in fns[0].args.args]
    if got != want:
        raise ValueError(f'hook {name!r} parameters {got} != {want}')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(m.split('.')[0] in DENY_IMPORTS for m in mods):
                raise ValueError(f'hook {name!r}: forbidden import')
        if isinstance(node, ast.Name) and node.id in DENY_CALLS:
            raise ValueError(f'hook {name!r}: forbidden name {node.id}')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr != '__name__':
            raise ValueError(f'hook {name!r}: dunder access')
    namespace = dict(vars(inspect.getmodule(original)))   # same globals the original sees
    exec(compile(tree, f'<hook {name}>', 'exec'), namespace)
    return namespace[name]

def materialize(cls, hooks, policy='declared'):
    """Return a subclass of cls whose listed methods run the hook code, falling back to the original on error."""
    hooks = {n: s for n, s in (hooks or {}).items() if s is not None}
    if not hooks:
        return cls
    allowed = hook_points(cls, policy)
    body = {}
    for name, source in hooks.items():
        if name not in allowed:
            raise ValueError(f'{cls.__name__}.{name} is not a hook point ({sorted(allowed)})')
        original = getattr(cls, name)
        fn = _compile(name, source, original)
        def guarded(self, *a, __fn=fn, __orig=original, __name=name, **k):
            try:
                return __fn(self, *a, **k)
            except Exception:
                failures = self.__dict__.setdefault('hook_failures', {})
                failures[__name] = failures.get(__name, 0) + 1
                return __orig(self, *a, **k)
        body[name] = functools.wraps(original)(guarded)
    return type(f'{cls.__name__}Hooked', (cls,), body)
