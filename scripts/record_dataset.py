# scripts/record_dataset.py
import os
import time
import soundfile as sf
import sounddevice as sd

OUT_DIR = os.path.join("data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
SR = 44100  # grabación a 44.1 kHz (luego re-muestrearemos a 22.05k)

PROMPTS = [
    "Hola, estoy grabando mi voz para el proyecto de clonador.",
    "Este es un ejemplo de frase con distinta entonación.",
    "La ingeniería requiere datos, pruebas y sentido común.",
    "Probando pausas y entonación en una frase corta."
]

def record_clip(path, duration=5):
    print(f"Grabando {duration}s -> {path}")
    audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, audio, SR)
    print("Guardado:", path)

if __name__ == "__main__":
    print("Consejos: graba en un lugar silencioso, micrófono estable.")
    time.sleep(0.5)
    for i, text in enumerate(PROMPTS, start=1):
        print("\nFrase a leer:")
        print(text)
        input("Presiona ENTER y empieza a hablar (después de 1s)...")
        time.sleep(1)
        out = os.path.join(OUT_DIR, f"utt_{i:03d}.wav")
        record_clip(out, duration=5)
    print("\nTerminó la grabación. Revisa data/raw/")
