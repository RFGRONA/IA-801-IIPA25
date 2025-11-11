# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['interfaz.py'],
    pathex=[],
    binaries=[],
<<<<<<< HEAD
    datas=[
        ('escudo.png', '.'),
        ('logo.png', '.')
    ],
    hiddenimports=['PIL._tkinter_finder'],
=======
    datas=[('escudo.png', '.'), ('logo.png', '.')],
    hiddenimports=[],
>>>>>>> oscar
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
<<<<<<< HEAD
    [],
    exclude_binaries=True,
=======
    a.binaries,
    a.datas,
    [],
>>>>>>> oscar
    name='interfaz',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
<<<<<<< HEAD
=======
    upx_exclude=[],
    runtime_tmpdir=None,
>>>>>>> oscar
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
<<<<<<< HEAD
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='interfaz',
)
=======
>>>>>>> oscar
