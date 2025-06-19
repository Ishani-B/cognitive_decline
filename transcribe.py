import os
import glob
import whisper

model = whisper.load_model("base")
input_root  = "audio_input"
output_root = "transcript_input"
subfolders  = ["cc", "cd"]

for sub in subfolders:
    in_dir  = os.path.join(input_root,  sub)
    out_dir = os.path.join(output_root, sub)
    os.makedirs(out_dir, exist_ok=True)
    
    for audio_path in glob.glob(os.path.join(in_dir, "*.*")):
        if not audio_path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")):
            continue
        result = model.transcribe(audio_path)
        text   = result["text"].strip()
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(out_dir, base + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"⏱  {sub}: {base} → {out_path}")

print("All done!")
