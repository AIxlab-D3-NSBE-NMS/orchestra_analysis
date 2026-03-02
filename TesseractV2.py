import cv2
import pytesseract
import re
import os
import csv
import numpy as np

# Caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

video_path = "2026-01-22_17-19-18-764360.mp4"
#video_path = "2026-01-22_17-19-15-226695.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir vídeo")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total frames no vídeo: {total_frames}")

num_frames_to_process = min(2000, total_frames)
print(f"Processando os primeiros {num_frames_to_process} frames...")

output_dir = "rois_detectadas"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "resultados.csv")

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame_Index", "Timestamp", "Texto_Timestamp", "Frame_Num", "Texto_Frame"])

def crop_to_content(img):
    # Encontrar coordenadas dos pixels brancos
    coords = cv2.findNonZero(img)

    if coords is None:
        return img  # evita erro se estiver vazio

    x, y, w, h = cv2.boundingRect(coords)
    cropped = cv2.bitwise_not(img[y:y+h, x:x+w])

    return cropped

# =====================================================
# PRÉ PROCESSAMENTO
# =====================================================
def preprocess(img):
    # Converter para grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aumentar escala para melhorar OCR
    gray = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

    # Aplicar CLAHE para contraste melhorado
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Blur leve para remover ruído
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # ==============================
    # THRESHOLD MANUAL (não OTSU)
    # ==============================

    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Forçar texto branco sobre fundo preto
    #thresh = cv2.bitwise_not(thresh)

    return thresh

# =====================================================
# OCR FUNÇÕES
# =====================================================
def ocr_timestamp(img):
    processed = crop_to_content(preprocess(img))
    padding = 10  # pixels
    processed = cv2.copyMakeBorder(
        processed,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    return pytesseract.image_to_string(processed, config=config).strip()


def ocr_frame(img):
    processed = preprocess(img)
    processed = crop_to_content(processed)
    padding = 15  # pixels
    processed = cv2.copyMakeBorder(
        processed,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    return pytesseract.image_to_string(processed, config=config).strip()


frame_index = 0
roi_saved = False

timestamps_list = []
frame_nums_list = []


cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # Escolher frame para começar 
frame_index = 1
while frame_index < num_frames_to_process:

    ret, frame = cap.read()
    if not ret:
        break

    # ROI fixa
    roi = frame[0:40, 0:120]
    roi_top = roi[0:12, :]
    roi_bottom = roi[15:28, :]  #15:28 para o ecra, 16:28 para a camera 

    # Escolher frame para visualizar em caso de debug 
    if frame_index == 681: #220
        roi_saved = False

    if not roi_saved:
        cv2.imwrite(os.path.join(output_dir, "roi_original.png"), roi)
        cv2.imwrite(os.path.join(output_dir, "roi_top_original.png"), roi_top)
        cv2.imwrite(os.path.join(output_dir, "roi_bottom_original.png"), roi_bottom)

        cv2.imwrite(os.path.join(output_dir, "roi_top_processed.png"), cv2.copyMakeBorder(
        crop_to_content(preprocess(roi_top)),
        10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=0
    ))
        cv2.imwrite(os.path.join(output_dir, "roi_bottom_processed.png"), cv2.copyMakeBorder(
        crop_to_content(preprocess(roi_bottom)),
        30, 30, 30, 30,
        cv2.BORDER_CONSTANT,
        value=0))

        print("ROI original e processada guardadas.")
        roi_saved = True

    # OCR
    timestamp_text = ocr_timestamp(roi_top)
    frame_text = ocr_frame(roi_bottom)

    # Regex
    time_match = re.search(r'\d+\.\d+', timestamp_text)
    frame_match = re.search(r'(\d+)', frame_text)

    timestamp = float(time_match.group(0)) if time_match else None
    frame_num = int(frame_match.group(1)) if frame_match else None

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_index, timestamp, timestamp_text, frame_num, frame_text])

    if timestamp is not None and frame_num is not None:
        timestamps_list.append(timestamp)
        frame_nums_list.append(frame_num)

    frame_index += 1

cap.release()

print(f"Processamento dos {num_frames_to_process} frames concluído.")
print(f"Resultados salvos em: {csv_file}")
