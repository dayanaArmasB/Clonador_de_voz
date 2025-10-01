# scripts/tts_zeroshot.py
import os, glob
import torch

# ✅ Solo agregamos XttsConfig como seguro
from TTS.tts.configs.xtts_config import XttsConfig

torch.serialization.add_safe_globals([XttsConfig])

# ✅ Parche universal para forzar weights_only=False
_real_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)

torch.load = _torch_load_patch


from TTS.api import TTS

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
OUT_PATH = os.path.join("data", "demo_dayana.wav")

def collect_refs(max_refs=6):
    refs = sorted(glob.glob(os.path.join("data","processed","*.wav")))
    return refs[:max_refs]

def main():
    refs = collect_refs()
    if not refs:
        raise SystemExit("No hay audios procesados en data/processed. Ejecuta preprocess.py primero.")
    tts = TTS(model_name=MODEL)  # ahora debería cargar bien
    text = "Hola, este es un demo de clonación de voz usando varias referencias."
    tts.tts_to_file(text=text, speaker_wav=refs, language="es", file_path=OUT_PATH)
    print("Demo generado:", OUT_PATH)

if __name__ == "__main__":
    main()

