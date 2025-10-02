# scripts/tts_zeroshot_interactive.py
"""
Script interactivo para síntesis zero-shot con Coqui XTTS.
Interfaz en español; permite seleccionar texto, idioma, emoción (interfaz), velocidad, # referencias y archivo de salida.

NOTAS:
- El parámetro 'emoción' se muestra en español y se mapea a etiquetas en inglés (internas). Algunos modelos/versión de TTS podrían no usar directamente esa etiqueta.
- El script usa tts.tts_to_file(...) con speaker_wav (lista) y language.
- Se aplica post-procesado de velocidad (time-stretch) si se indica speed != 1.0 usando librosa.

Requisitos: ya deberías tener instaladas las dependencias TTS, torch, librosa, soundfile, numpy.

Uso:
    python scripts/tts_zeroshot_interactive.py

"""
import os
import glob
import sys
import tempfile
import shutil

# Parches necesarios para cargar modelos XTTS en algunos entornos
import torch
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])

_real_torch_load = torch.load
def _torch_load_patch(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _real_torch_load(*args, **kwargs)
torch.load = _torch_load_patch

from TTS.api import TTS
import librosa
import soundfile as sf

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_OUT = os.path.join("data", "demo_dayana_interactive.wav")
DEFAULT_MAX_REFS = 12


def collect_refs(max_refs=12):
    refs = sorted(glob.glob(os.path.join("data", "processed", "*.wav")))
    return refs[:max_refs]


def time_stretch_file(in_path, out_path, speed=1.0, sr=22050):
    if abs(speed - 1.0) < 1e-6:
        # copiar directamente
        shutil.copy(in_path, out_path)
        return
    try:
        y, _sr = librosa.load(in_path, sr=sr)
        # librosa.effects.time_stretch requiere frames hop/phase pero funciona bien para cambios moderados
        y_st = librosa.effects.time_stretch(y, rate=speed)
        sf.write(out_path, y_st, sr, subtype='PCM_16')
    except Exception as e:
        print("[WARN] No se pudo aplicar time-stretching:", e)
        shutil.copy(in_path, out_path)


def choose_emotion():
    options = [
        ("Neutro", "neutral"),
        ("Alegre", "cheerful"),
        ("Tranquilo", "calm"),
        ("Serio", "serious"),
        ("Enfadado", "angry"),
        ("Triste", "sad"),
    ]
    print("Selecciona emoción (interfaz en español):")
    for i, (esp, _) in enumerate(options, start=1):
        print(f"  {i}. {esp}")
    sel = input(f"Elige 1-{len(options)} [1]: ").strip() or "1"
    try:
        idx = int(sel)
        if not (1 <= idx <= len(options)):
            idx = 1
    except:
        idx = 1
    return options[idx-1][1]


def main():
    print("=== TTS Zero-Shot interactivo (Coqui XTTS) ===")

    text = input("Texto a sintetizar (ENTER para usar texto por defecto): \n")
    if not text.strip():
        text = "Hola, este es un demo de clonación de voz."

    lang = input("Idioma para pronunciación [es]: ").strip() or "es"

    emotion = choose_emotion()

    speed_raw = input("Velocidad (float, 1.0 = normal) [1.0]: ").strip() or "1.0"
    try:
        speed = float(speed_raw)
        if speed <= 0:
            speed = 1.0
    except:
        speed = 1.0

    temp_raw = input("Temperature/aleatoriedad (float 0.0-1.5) [0.7]: ").strip() or "0.4"
    try:
        temperature = float(temp_raw)
    except:
        temperature = 0.7

    max_refs_raw = input(f"Máximo de referencias a usar (archivos en data/processed) [{DEFAULT_MAX_REFS}]: ").strip() or str(DEFAULT_MAX_REFS)
    try:
        max_refs = int(max_refs_raw)
        if max_refs <= 0:
            max_refs = DEFAULT_MAX_REFS
    except:
        max_refs = DEFAULT_MAX_REFS

    out = input(f"Archivo de salida (path) [{DEFAULT_OUT}]: ").strip() or DEFAULT_OUT
    out = os.path.abspath(out)

    print("\nRecopilando referencias procesadas...")
    refs = collect_refs(max_refs)
    if not refs:
        print("No hay audios en data/processed/. Ejecuta preprocess.py o copia archivos allí.")
        sys.exit(1)

    print(f"Usando {len(refs)} referencias (máx {max_refs}). Sintetizando...")

    # Cargar modelo y sintetizar a archivo temporal
    tts = TTS(model_name=MODEL)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        # Nota: no pasamos emotion/temperature directamente porque la API pública puede no aceptarlos.
        # Guardamos el resultado en tmp_path y luego aplicamos post-procesado (speed) si se pidió.
        tts.tts_to_file(
            text=text,
            speaker_wav=refs,
            language=lang,
            file_path=tmp_path
        )

        # Post-procesado: aplicar velocidad si es distinto de 1.0
        if abs(speed - 1.0) > 1e-6:
            print("Aplicando ajuste de velocidad (time-stretch)... Esto puede tardar unos segundos.")
            time_stretched_path = out
            time_stretch_file(tmp_path, time_stretched_path, speed=speed, sr=22050)
        else:
            shutil.move(tmp_path, out)

        print("✅ Generado:", out)
        print(f"(Emoción seleccionada: {emotion}; temperature solicitada: {temperature})")
        print("Nota: Emoción/temperature se muestran como guía; si quieres que TTS use directamente estos parámetros, podemos adaptar el pipeline a un modelo o a finetuning.")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


if __name__ == "__main__":
    main()


# --------------------------------------------------
# LISTA DE FRASES (MIXTO) — ÚSALAS PARA GRABAR (≈10s c/u)
# --------------------------------------------------
# 1) Personal - Presentación clara
# "Hola, soy Dayana. Estoy colaborando en un proyecto de clonación de voz."
#
# 2) Personal - Conversacional
# "¡Hola! Me alegra que estés aquí. Hoy voy a leer unas frases para mi dataset." 
#
# 3) Personal - Emocional alegre
# "¡Qué emocionante ver cómo la tecnología nos permite crear cosas nuevas!"
#
# 4) Narrador - Corporativo
# "La Ingeniería de Sistemas integra datos, procesos y personas para solucionar problemas reales."
#
# 5) Narrador - Reflexivo
# "Cuando observamos un sistema, debemos analizar entradas, salidas y las relaciones internas."
#
# 6) Personal - Susurro / cercano
# "Escucha con atención, cada palabra importa cuando estás construyendo una historia." 
#
# 7) Personal - Enfática / seria
# "Por favor, asegúrate de leer cada instrucción con precisión y calma." 
#
# 8) Narrador - Descriptivo
# "En un día soleado, la gente caminaba por la plaza mientras los vendedores ofrecían sus productos." 
#
# 9) Personal - Pregunta / interrogativa
# "¿Qué te motivó a estudiar ingeniería? Cuéntame tu experiencia en pocas palabras." 
#
# 10) Narrador - Cierre / despedida
# "Y así concluye nuestro breve experimento; gracias por escuchar y por participar." 
#
# Recomendaciones de grabación:
# - Graba cada frase en silencio, con el micrófono a una distancia constante (20-30 cm).
# - Deja 0.5-1s de silencio al principio y al final; preprocess.py recortará silencios si es necesario.
# - Si vas a grabar 10s por frase, puedes leer la frase y añadir 2-4 oraciones naturales para completar la duración.
# - Guarda los archivos en 'data/raw/' y luego ejecuta 'python scripts/preprocess.py'
# --------------------------------------------------
