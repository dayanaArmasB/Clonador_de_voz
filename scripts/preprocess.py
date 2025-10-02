# scripts/preprocess.py
import os
import glob
import json
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# CONFIG
RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")
SR = 22050
HP_CUTOFF = 80          # High-pass cutoff (Hz) para quitar rumble
TOP_DB = 30             # Umbral para librosa.effects.split
MIN_DURATION = 0.5      # Segmentar silencios: descartamos segmentos < 0.5s
TARGET_RMS_DB = -25.0   # RMS objetivo en dBFS (ajusta según preferencia)

os.makedirs(OUT_DIR, exist_ok=True)

def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='highpass')
    return b, a

def apply_highpass(y, sr, cutoff=HP_CUTOFF):
    try:
        b, a = butter_highpass(cutoff, sr, order=6)
        y = filtfilt(b, a, y)
    except Exception as e:
        # fallback si filtfilt falla por razones de longitud
        pass
    return y

def remove_dc_offset(y):
    return y - np.mean(y) if y.size else y

def vad_concatenate(y, sr, top_db=TOP_DB, min_duration=MIN_DURATION):
    intervals = librosa.effects.split(y, top_db=top_db)
    pieces = []
    for start, end in intervals:
        dur = (end - start) / sr
        if dur >= min_duration:
            pieces.append(y[start:end])
    if not pieces:
        return np.array([])
    return np.concatenate(pieces)

def rms_db(y):
    # rms in linear, convert to dBFS (0 dBFS = 1.0 peak)
    rms = np.sqrt(np.mean(np.square(y))) if y.size else 0.0
    if rms <= 1e-9:
        return -999.0
    return 20 * np.log10(rms)

def normalize_rms(y, target_db=TARGET_RMS_DB):
    current_db = rms_db(y)
    if current_db < -900:
        return y
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    return y * gain

def process_file(in_path, out_path):
    # Load using soundfile to preserve fidelity
    data, orig_sr = sf.read(in_path, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)   # stereo -> mono
    # Resample if needed
    if orig_sr != SR:
        data = librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=SR)

    # Remove DC offset
    data = remove_dc_offset(data)

    # High-pass to remove rumble
    data = apply_highpass(data, SR, cutoff=HP_CUTOFF)

    # VAD: keep voiced regions only
    data = vad_concatenate(data, SR, top_db=TOP_DB, min_duration=MIN_DURATION)
    if data.size == 0:
        print(f"[WARN] {os.path.basename(in_path)} -> no speech detected, skipping.")
        return None

    # Normalize RMS to target
    data = normalize_rms(data, target_db=TARGET_RMS_DB)

    # Final peak-limit to avoid clipping
    peak = np.max(np.abs(data)) if data.size else 1.0
    if peak > 0.99:
        data = data / peak * 0.98

    # Save
    sf.write(out_path, data.astype(np.float32), SR, subtype='PCM_16')

    # Metadata
    meta = {
        "filename": os.path.basename(out_path),
        "orig_file": os.path.basename(in_path),
        "sr": SR,
        "duration_s": float(len(data) / SR),
        "rms_db": float(rms_db(data)),
        "peak": float(np.max(np.abs(data)))
    }
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Procesado:", os.path.basename(out_path))
    return meta

if __name__ == "__main__":
    wavs = sorted(glob.glob(os.path.join(RAW_DIR, "*.wav")))
    if not wavs:
        print("No hay archivos en data/raw. Graba o copia tus WAV allí.")
    else:
        for w in wavs:
            out = os.path.join(OUT_DIR, os.path.basename(w))
            process_file(w, out)
        print("Preprocesamiento completado. Archivos en data/processed/")
