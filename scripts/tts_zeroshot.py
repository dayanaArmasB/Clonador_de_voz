import os
import glob
import argparse
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
DEFAULT_OUT = os.path.join("data", "demo_dayana.wav")

def collect_refs(max_refs=6):
    refs = sorted(glob.glob(os.path.join("data", "processed", "*.wav")))
    return refs[:max_refs]

def main():
    parser = argparse.ArgumentParser(description="Zero-shot TTS con Coqui XTTS")
    parser.add_argument("--text", type=str, default="Hola, este es un demo de clonación de voz.Me llamo Raúl Armas, vivo en San Miguel en Fortunato Quezada 109.",
                        help="Texto a sintetizar")
    parser.add_argument("--lang", type=str, default="es",
                        help="Idioma para la pronunciación (ej: es, en, fr)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="Ruta del archivo de salida .wav")
    parser.add_argument("--max_refs", type=int, default=6,
                        help="Número máximo de referencias de voz a usar")

    args = parser.parse_args()

    refs = collect_refs(args.max_refs)
    if not refs:
        raise SystemExit("No hay audios procesados en data/processed. Ejecuta preprocess.py primero.")

    tts = TTS(model_name=MODEL)

    tts.tts_to_file(
        text=args.text,
        speaker_wav=refs,
        language=args.lang,
        file_path=args.out
    )

    print("✅ Audio generado:", args.out)

if __name__ == "__main__":
    main()


