# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts/run_app.py'],
    pathex=['src'],
    binaries=[],
    datas=[('models/nemotron_diarization/Nemotron-3-Diarization.onnx', 'models/nemotron_diarization'), ('models/sensevoice_small/am.mvn', 'models/sensevoice_small'), ('models/sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model', 'models/sensevoice_small'), ('models/sensevoice_small/config.yaml', 'models/sensevoice_small'), ('models/sensevoice_small/configuration.json', 'models/sensevoice_small'), ('models/sensevoice_small/SenseVoiceSmall.onnx', 'models/sensevoice_small'), ('models/sensevoice_small/SenseVoiceSmall.onnx.data', 'models/sensevoice_small'), ('src/diarizeflow/app/frontend/web', 'diarizeflow/app/frontend/web'), ('config.json', '.')],
    hiddenimports=['PySide6', 'onnxruntime', 'websockets', 'sentencepiece', 'sounddevice', 'soundcard', 'cffi', 'soundfile', 'kaldi_native_fbank', 'uvicorn', 'fastapi', 'onnx', 'onnxconverter_common'],
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
    [],
    exclude_binaries=True,
    name='DiarizeFlow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DiarizeFlow',
)
