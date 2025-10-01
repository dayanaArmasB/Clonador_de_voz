# scripts/preprocess.py
import os, glob
import librosa
import soundfile as sf
import numpy as np

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")
SR = 22050
os.makedirs(OUT_DIR, exist_ok=True)

def trim_and_normalize(y):
    yt, _ = librosa.effects.trim(y, top_db=30)
    peak = np.max(np.abs(yt)) if yt.size else 1.0
    if peak > 0:
        yt = yt / peak * 0.9
    return yt

def process_file(in_path, out_path):
    y, sr = librosa.load(in_path, sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = trim_and_normalize(y)
    sf.write(out_path, y, SR, subtype='PCM_16')
    print("Procesado:", os.path.basename(out_path))

if __name__ == "__main__":
    wavs = sorted(glob.glob(os.path.join(RAW_DIR, "*.wav")))
    if not wavs:
        print("No hay archivos en data/raw. Graba o copia tus WAV allí.")
    for w in wavs:
        out = os.path.join(OUT_DIR, os.path.basename(w))
        process_file(w, out)
    print("Preprocesamiento completado. Archivos en data/processed/")
