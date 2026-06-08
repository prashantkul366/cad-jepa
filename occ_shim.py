"""
occ_shim.py  (v9 — dunder fix + hasattr fix)
Bridges OCC.Core.* -> OCP.* and stubs OCC.Extend.*.
"""

import importlib
import sys
import types
import warnings


# =============================================================================
# _Stub
# =============================================================================

class _Stub:
    def __getattr__(self, _):    return _Stub()
    def __call__(self, *a, **k): return _Stub()
    def __iter__(self):          return iter([])
    def __bool__(self):          return False
    def __repr__(self):          return '<occ_shim._Stub>'


# =============================================================================
# _StaticProxy
# =============================================================================

class _StaticProxy:
    def __init__(self, ocp_class, label=''):
        object.__setattr__(self, '_cls',   ocp_class)
        object.__setattr__(self, '_label', label)

    def __getattr__(self, name):
        cls = object.__getattribute__(self, '_cls')
        lbl = object.__getattribute__(self, '_label')
        for cand in (name + '_s', name):
            if hasattr(cls, cand):
                return getattr(cls, cand)
        raise AttributeError(
            f"occ_shim: {lbl or cls.__name__} has no '{name}' or '{name}_s'")


# =============================================================================
# Universal compat module builder
# =============================================================================

def _make_compat_module(occ_name, ocp_mod):
    mod = types.ModuleType(occ_name)
    mod.__package__ = 'OCC.Core'

    # Set standard module dunders explicitly so they never hit __getattr__
    mod.__spec__   = None
    mod.__file__   = getattr(ocp_mod, '__file__', None)
    mod.__loader__ = getattr(ocp_mod, '__loader__', None)
    mod.__path__   = None   # not a package
    mod.__doc__    = getattr(ocp_mod, '__doc__', None)

    # Copy all non-dunder OCP attributes into __dict__
    for a in dir(ocp_mod):
        if not a.startswith('__'):
            try: setattr(mod, a, getattr(ocp_mod, a))
            except Exception: pass

    _ocp  = ocp_mod
    _name = occ_name

    def _getattr(attr, _ocp=_ocp, _name=_name):
        # FIX 1: never stub dunder attrs — raise AttributeError so Python
        # handles them correctly (hasattr returns False, inspect works, etc.)
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)

        # 1. Direct on OCP module
        try: return getattr(_ocp, attr)
        except AttributeError: pass

        # 2. Search OCP classes for attr_s / attr static methods
        for cls_name in dir(_ocp):
            if cls_name.startswith('_'): continue
            cls_obj = getattr(_ocp, cls_name, None)
            if not isinstance(cls_obj, type): continue
            for cand in (attr + '_s', attr):
                if hasattr(cls_obj, cand):
                    return getattr(cls_obj, cand)

        # 3. Stub with warning
        warnings.warn(
            f"occ_shim: '{attr}' not found in OCP counterpart of '{_name}'. "
            f"Stubbed — may cause issues if called during reconstruction.",
            stacklevel=2)
        return _Stub()

    mod.__getattr__ = _getattr
    return mod


# =============================================================================
# Patch builders
# FIX 2: use  `name not in mod.__dict__`  instead of  `not hasattr(mod, name)`
# hasattr triggers __getattr__ which returns _Stub() (no AttributeError),
# so hasattr always returns True and the correct value was never being set.
# =============================================================================

def _free_func_patch(occ_name, ocp_module, occ_prefix, ocp_class_name,
                     proxy_attrs=None):
    def _build():
        ocp = importlib.import_module(ocp_module)
        mod = _make_compat_module(occ_name, ocp)

        if proxy_attrs:
            for attr, cls_name in proxy_attrs.items():
                if attr not in mod.__dict__:          # ← FIX 2
                    ocp_cls = getattr(ocp, cls_name, None)
                    if ocp_cls:
                        setattr(mod, attr, _StaticProxy(ocp_cls, f'{occ_name}.{attr}'))

        ocp_cls = getattr(ocp, ocp_class_name, None)
        if ocp_cls:
            pfx = occ_prefix; cls = ocp_cls
            orig = getattr(mod, '__getattr__', None)
            def _ga(name, _p=pfx, _c=cls, _orig=orig):
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                if name.startswith(_p):
                    method = name[len(_p):]
                    for cand in (method + '_s', method):
                        if hasattr(_c, cand): return getattr(_c, cand)
                if _orig: return _orig(name)
                raise AttributeError(name)
            mod.__getattr__ = _ga
        return mod
    return _build


def _patch_breptools():
    ocp = importlib.import_module('OCP.BRepTools')
    mod = _make_compat_module('OCC.Core.BRepTools', ocp)
    BT  = getattr(ocp, 'BRepTools', None)

    def _w(shape, filename, append=False):
        if BT and hasattr(BT, 'Write_s'):
            try:    BT.Write_s(shape, filename, append)
            except TypeError: BT.Write_s(shape, filename)
    def _r(shape_ref, filename, builder=None):
        if BT and hasattr(BT, 'Read_s'):
            try:    BT.Read_s(shape_ref, filename, builder)
            except TypeError: BT.Read_s(shape_ref, filename)
    def _c(shape):
        if BT and hasattr(BT, 'Clean_s'): BT.Clean_s(shape)

    for name, fn in [('breptools_Write', _w),
                     ('breptools_Read',  _r),
                     ('breptools_Clean', _c)]:
        if name not in mod.__dict__:                  # ← FIX 2
            setattr(mod, name, fn)
    return mod


_COMPAT_PATCHES = {
    'OCC.Core.BRepTools':   _patch_breptools,
    'OCC.Core.BRepGProp':   _free_func_patch('OCC.Core.BRepGProp',  'OCP.BRepGProp',
                                              'brepgprop_',  'BRepGProp',
                                              proxy_attrs={'brepgprop': 'BRepGProp'}),
    'OCC.Core.BRepLib':     _free_func_patch('OCC.Core.BRepLib',    'OCP.BRepLib',
                                              'breplib_',    'BRepLib',
                                              proxy_attrs={'breplib': 'BRepLib'}),
    'OCC.Core.BRepAlgo':    _free_func_patch('OCC.Core.BRepAlgo',   'OCP.BRepAlgo',
                                              'brepalgo_',   'BRepAlgo',
                                              proxy_attrs={'brepalgo': 'BRepAlgo'}),
    'OCC.Core.BRepBndLib':  _free_func_patch('OCC.Core.BRepBndLib', 'OCP.BRepBndLib',
                                              'brepbndlib_', 'BRepBndLib'),
    'OCC.Core.TopExp':      _free_func_patch('OCC.Core.TopExp',     'OCP.TopExp',
                                              'topexp_',     'TopExp'),
    'OCC.Core.TopoDS':      _free_func_patch('OCC.Core.TopoDS',     'OCP.TopoDS',
                                              'topods_',     'TopoDS'),
    'OCC.Core.GeomAPI':     _free_func_patch('OCC.Core.GeomAPI',    'OCP.GeomAPI',
                                              'geomapi_',    'GeomAPI'),
    'OCC.Core.GeomConvert': _free_func_patch('OCC.Core.GeomConvert','OCP.GeomConvert',
                                              'geomconvert_','GeomConvert'),
}


# =============================================================================
# OCC.Extend stubs
# =============================================================================

def _write_stl_file_real(a_shape, filename, mode="ascii",
                         linear_deflection=0.9, angular_deflection=0.5):
    """
    Real STL export using OCP (replaces pythonocc OCC.Extend.DataExchange.write_stl_file).
    Called by cadlib/visualize.py:CADsolid2pc — must actually create the file.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    mesh = BRepMesh_IncrementalMesh(a_shape, linear_deflection, False, angular_deflection)
    mesh.Perform()
    writer = StlAPI_Writer()
    try:
        writer.ASCIIMode = (mode == "ascii")
    except AttributeError:
        pass  # some OCP builds expose this differently
    writer.Write(a_shape, filename)


def _extend_stub(fullname):
    mod = types.ModuleType(fullname)
    mod.__path__ = []; mod.__package__ = fullname
    mod.__spec__ = None; mod.__file__ = None
    for name, val in {
        'TopologyExplorer': _Stub, 'WireExplorer': _Stub,
        'make_edge': _Stub(), 'make_wire': _Stub(), 'make_face': _Stub(),
        'make_extrusion': _Stub(), 'make_vertex': _Stub(),
        'get_sorted_hlr_edges': _Stub(),
        'write_step_file': _Stub(), 'read_step_file': _Stub(),
        'write_stl_file': _write_stl_file_real,  # REAL — cadlib uses this
        'write_iges_file': _Stub(), 'read_iges_file': _Stub(),
        'export_shape_to_svg': _Stub(),
        'DataExchange': _Stub, 'STEP_Export': _Stub(), 'STL_Export': _Stub(),
    }.items():
        setattr(mod, name, val)
    def _ga(name):
        if name.startswith('__') and name.endswith('__'): raise AttributeError(name)
        return _Stub()
    mod.__getattr__ = _ga
    return mod


# =============================================================================
# Meta-path hook
# =============================================================================

class _OCCBridge:

    def find_module(self, name, path=None):
        if name == 'OCC' or name.startswith('OCC.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        if fullname == 'OCC':
            mod = types.ModuleType('OCC')
            mod.__path__ = []; mod.__package__ = 'OCC'; mod.__spec__ = None
            sys.modules['OCC'] = mod; return mod

        if fullname == 'OCC.Core':
            mod = types.ModuleType('OCC.Core')
            mod.__path__ = []; mod.__package__ = 'OCC.Core'; mod.__spec__ = None
            sys.modules['OCC.Core'] = mod; return mod

        if fullname == 'OCC.Extend' or fullname.startswith('OCC.Extend.'):
            mod = _extend_stub(fullname)
            sys.modules[fullname] = mod; return mod

        if fullname in _COMPAT_PATCHES:
            mod = _COMPAT_PATCHES[fullname]()
            sys.modules[fullname] = mod; return mod

        if fullname.startswith('OCC.Core.'):
            ocp_name = 'OCP.' + fullname[len('OCC.Core.'):]
        elif fullname.startswith('OCC.'):
            ocp_name = 'OCP.' + fullname[len('OCC.'):]
        else:
            raise ImportError(fullname)

        try:
            ocp_mod = importlib.import_module(ocp_name)
            compat  = _make_compat_module(fullname, ocp_mod)
            sys.modules[fullname] = compat; return compat
        except ImportError:
            raise ImportError(
                f"occ_shim: '{ocp_name}' not in cadquery-ocp "
                f"(resolving '{fullname}')")


def install():
    if not any(isinstance(h, _OCCBridge) for h in sys.meta_path):
        sys.meta_path.insert(0, _OCCBridge())

install()


# =============================================================================
# Diagnostic
# =============================================================================

if __name__ == '__main__':
    import os
    print("Importing cadquery..."); import cadquery; install()

    probes = [
        'OCC.Core.BRepMesh', 'OCC.Core.StlAPI', 'OCC.Core.BRep',
        'OCC.Core.BRepBuilderAPI', 'OCC.Core.BRepTools',
        'OCC.Core.BRepGProp', 'OCC.Core.BRepLib', 'OCC.Core.BRepBndLib',
        'OCC.Core.TopTools', 'OCC.Core.TopExp', 'OCC.Core.TopoDS',
        'OCC.Core.TopAbs', 'OCC.Core.gp',
        'OCC.Extend.TopologyUtils', 'OCC.Extend.ShapeFactory',
    ]
    print("\nBridge + stubs (should be silent):"); print("=" * 50)
    for m in probes:
        try:
            importlib.import_module(m); print(f"  ✓  {m}")
        except ImportError as e:
            print(f"  ✗  {m}: {e}")

    from OCC.Core.BRepTools  import breptools_Write
    from OCC.Core.BRepGProp  import brepgprop, brepgprop_LinearProperties
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.TopTools   import TopTools_ListIteratorOfListOfShape
    from OCC.Core.gp         import gp_Pnt
    from OCC.Extend.TopologyUtils import TopologyExplorer

    print(f"\n  breptools_Write:                  callable={callable(breptools_Write)}")
    print(f"  brepgprop:                        type={type(brepgprop).__name__}")
    print(f"  brepgprop_LinearProperties:       callable={callable(brepgprop_LinearProperties)}")
    print(f"  brepbndlib_Add:                   callable={callable(brepbndlib_Add)}")
    print(f"  TopTools_ListIteratorOfListOfShape:type={type(TopTools_ListIteratorOfListOfShape).__name__}")
    print(f"  gp_Pnt:                           type={type(gp_Pnt).__name__}")

    T2CAD = r'C:\Users\ve00yn139\OneDrive - YAMAHA MOTOR CO., LTD\Desktop\CAD Researcg\Text2CAD'
    if os.path.isdir(T2CAD): sys.path.insert(0, T2CAD)

    print("\nCadSeqProc import:")
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")   # suppress stubs warnings during import
        try:
            from CadSeqProc.cad_sequence import CADSequence
            print("  ✓  CadSeqProc.CADSequence importable")
            print("\n✓  ALL CLEAR — run_cadquery_eval.py is unblocked.")
        except Exception as e:
            print(f"  ✗  {type(e).__name__}: {e}")