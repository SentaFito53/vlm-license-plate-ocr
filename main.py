import os
import re              # regular expresion utk extract output model
import csv
import jiwer           # untuk hitung metrik error (CER)
import lmstudio as lms

# KONFIGURASI PATH
BASE = r"D:\SENTA\vlm-license-plate-ocr-aas\Indonesian License Plate Recognition Dataset"
IMG_DIR = os.path.join(BASE, "images", "test")
LABEL_DIR = os.path.join(BASE, "labels", "test")
CLASSES_FILE = os.path.join(BASE, "classes.names")

# FUNGSI UNTUK MENCARI NAMA FILE CSV YANG BELUM DIPAKAI
def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename

    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

OUTPUT_CSV = get_unique_filename("result_ocr_plat.csv")

# LOAD CLASS NAMES
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]

def build_ground_truth(label_path):
    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            char = classes[class_id]
            detections.append((x_center, char))
    detections.sort(key=lambda d: d[0])
    return "".join(d[1] for d in detections)

def extract_plate(text):
    text = str(text).upper()
    pattern = r'\b([A-Z]{1,2})\s?(\d{1,4})\s?([A-Z]{0,3})\b'
    matches = re.findall(pattern, text)
    for huruf_depan, angka, huruf_belakang in matches:
        if angka:
            plate = f"{huruf_depan}{angka}{huruf_belakang}"
            return re.sub(r'\s+', '', plate)
    return re.sub(r'\s+', '', text.strip())

model = lms.llm("llava-llama-3-8b-v1_1")

image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# BUKA FILE CSV DI AWAL, TULIS PER BARIS
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    f.write("sep=\t\n")  # <-- baris ini bikin Excel otomatis kenali delimiter tab
    writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "CER_score"], delimiter="\t")
    writer.writeheader()

    total_cer = 0
    count = 0

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(LABEL_DIR, base_name + ".txt")

        if not os.path.exists(label_path):
            print(f"[SKIP] Label tidak ditemukan untuk {img_file}")
            continue

        ground_truth = build_ground_truth(label_path)
        if not ground_truth:
            print(f"[SKIP] Ground truth kosong untuk {img_file}")
            continue

        img_path = os.path.join(IMG_DIR, img_file)
        image_handle = lms.prepare_image(img_path)

        chat = lms.Chat(
            "You are an OCR system. Your ONLY task is to read the license plate number in the image. "
            "Do not describe the image. Do not mention location, background, or context. "
            "Do not explain anything. Output ONLY the license plate number characters, nothing else."
        )
        chat.add_user_message(
            "What is the license plate number shown in this image? Respond only with the plate number.",
            images=[image_handle]
        )

        prediction_raw = model.respond(chat, config={"temperature": 0.0, "maxTokens": 15})
        prediction_clean = extract_plate(str(prediction_raw))

        measures = jiwer.process_characters(ground_truth, prediction_clean)
        S, D, I, N = measures.substitutions, measures.deletions, measures.insertions, len(ground_truth)
        cer_score = (S + D + I) / N if N > 0 else 0

        # LANGSUNG TULIS KE CSV & FLUSH
        writer.writerow({
            "image": img_file,
            "ground_truth": ground_truth,
            "prediction": prediction_clean,
            "CER_score": f"{cer_score:.4f}".replace(".", ",")
        })
        f.flush()

        total_cer += cer_score
        count += 1

        print(f"[{count}] {img_file:20s} | GT: {ground_truth:15s} | Pred: {prediction_clean:15s} | CER: {cer_score:.4f}")

    if count > 0:
        print(f"\nTotal gambar diproses : {count}")
        print(f"Rata-rata CER         : {total_cer / count:.4f}")
        print(f"Hasil disimpan di     : {OUTPUT_CSV}")
