# scripts/record_dataset.py
import os
import time
import soundfile as sf
import sounddevice as sd

# =============================
# CONFIGURACIÓN PRINCIPAL
# =============================
OUT_DIR = os.path.join("data", "raw")
DEFAULT_SR = 44100
DEFAULT_DURATION = 8
# =============================

os.makedirs(OUT_DIR, exist_ok=True)

# Frases por defecto si el usuario no quiere escribir las suyas
DEFAULT_PROMPTS = [
    "Hola, estoy grabando mi voz para el proyecto de clonador.",
    "Este es un ejemplo de frase con distinta entonación.",
    "La ingeniería requiere datos, pruebas y sentido común.",
    "Probando pausas y entonación en una frase corta.",
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "¿Cómo estás hoy? Espero que muy bien.",
    "La lluvia en Sevilla es una maravilla.",
    "Me gusta programar en Python y aprender cosas nuevas.",
]

def record_clip(path, duration, sr):
    print(f"🎙️ Grabando {duration}s -> {path}")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, audio, sr)
    print(f"✅ Guardado: {path}")

if __name__ == "__main__":
    print("======================================")
    print("🎧 Grabador de Dataset de Voz")
    print("Consejos: lugar silencioso, micrófono estable.")
    print("======================================\n")

    # Duración personalizada
    try:
        duration = int(input(f"⏱️ Duración por grabación en segundos (Default {DEFAULT_DURATION}): ") or DEFAULT_DURATION)
    except ValueError:
        duration = DEFAULT_DURATION

    # ¿Usar frases predeterminadas o personalizadas?
    use_custom = input("📝 ¿Quieres escribir tus frases personalizadas? (s/n): ").strip().lower() == "s"

    if use_custom:
        prompts = []
        print("\n✍️ Escribe tus frases (ENTER vacío para terminar):")
        while True:
            line = input("Frase: ").strip()
            if not line:
                break
            prompts.append(line)
        if not prompts:
            print("⚠️ No escribiste ninguna. Usando frases por defecto.")
            prompts = DEFAULT_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    # Proceso de grabación
    for i, text in enumerate(prompts, start=1):
        while True:  # Permite repetir si no quedó bien
            print(f"\n📢 Frase {i}/{len(prompts)}:")
            print(text)
            input(f"Presiona ENTER y empieza a hablar (comienza después de 1s)...")
            time.sleep(1)

            out = os.path.join(OUT_DIR, f"utt_{i:03d}.wav")
            record_clip(out, duration, DEFAULT_SR)

            retry = input("¿Quieres repetir esta grabación? (s/n): ").strip().lower()
            if retry != "s":
                break

    print("\n✅ ¡Grabación finalizada! Revisa los archivos en data/raw/")
