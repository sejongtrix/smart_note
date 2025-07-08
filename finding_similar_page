import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import cosine_similarity
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
import csv

print("🔧 모델 로딩 중...")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]) 
model.eval()
print("✅ 모델 로딩 완료")

# PDF 로드
print("📄 PDF 파일 로드 중...")
doc = fitz.open("test1.pdf")
print(f"✅ PDF 로드 완료 - 총 {len(doc)} 페이지")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 로그 파일 설정
output_path = "slide_similarity_log.csv"
csv_file = open(output_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["seconds", "page", "similarity"])
print(f"📝 CSV 로그 파일 초기화 완료: {output_path}")

# MSE 유사도 측정 함수
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# 프레임 → feature 추출
def extract_feature(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = model(tensor).squeeze()
    return feature

# 영상 파일 로딩
print("🎞️ 영상 로딩 중...")
cap = cv2.VideoCapture("test1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✅ 영상 로딩 완료 - FPS: {fps:.2f}")
frame_interval = int(fps * 1)

# 상태 변수들
last_slide = None
last_best_page = 0
prev_sim = None
frame_count = 0

# 하이퍼파라미터
mse_threshold = 500
sim_drop_threshold = 0.01
max_search_range = 10

print("🚀 분석 시작")

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 영상 끝")
        break

    frame_time = frame_count / fps

    if frame_count % frame_interval == 0:
        #print(f"\n🧭 프레임 {frame_count} (시간: {frame_time:.2f}초) 분석 중...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))

        if last_slide is None:
            print("🆕 첫 슬라이드로 설정")
            last_slide = gray
        else:
            diff = mse(gray, last_slide)
            #print(f"🔍 MSE 차이: {diff:.2f}")

            if diff > mse_threshold:
                print("📈 슬라이드 변경 감지됨 → 탐색 시작")
                frame_sim_results = []

                for offset in range(-int(2 * fps), int(2 * fps) + 1, frame_interval):
                    pos = frame_count + offset
                    if pos < 0 or pos >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret2, temp_frame = cap.read()
                    if not ret2:
                        continue

                    frame_rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                    feat1 = extract_feature(frame_rgb)

                    # 기본 후보 페이지 설정
                    candidate_range = [-2, -1, 0, 1, 2]
                    candidates = [last_best_page + i for i in candidate_range if 0 <= last_best_page + i < len(doc)]

                    best_page = last_best_page
                    max_sim = -1

                    print(f"🔎 1차 탐색: 후보 페이지 {candidates}")
                    for i in candidates:
                        page = doc[i]
                        pix = page.get_pixmap(dpi=300)
                        img_bytes = pix.tobytes("ppm")
                        feat2 = extract_feature(Image.open(io.BytesIO(img_bytes)))
                        sim = cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

                        print(f"    페이지 {i} 유사도: {sim:.4f}")
                        if sim > max_sim:
                            max_sim = sim
                            best_page = i
                            best_feat2 = feat2

                    # 유사도 급락 시 재탐색
                    if prev_sim is not None and (prev_sim - max_sim) >= sim_drop_threshold:
                        print(f"⚠️ 유사도 하락 감지 (이전: {prev_sim:.4f} → 현재: {max_sim:.4f}) → 2차 탐색")
                        expanded_range = list(range(-5, 6))
                        expanded_candidates = [last_best_page + i for i in expanded_range if 0 <= last_best_page + i < len(doc)]

                        print(f"🔍 2차 탐색: 후보 페이지 {expanded_candidates}")
                        for i in expanded_candidates:
                            page = doc[i]
                            pix = page.get_pixmap(dpi=300)
                            img_bytes = pix.tobytes("ppm")
                            feat2 = extract_feature(Image.open(io.BytesIO(img_bytes)))
                            sim = cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

                            print(f"    [2차] 페이지 {i} 유사도: {sim:.4f}")
                            if sim > max_sim:
                                max_sim = sim
                                best_page = i
                                best_feat2 = feat2

                    result_time = pos / fps
                    minutes = int(result_time // 60)
                    seconds = int(result_time % 60)
                    print(f"✅ [결과] {minutes}분 {seconds}초: 페이지 {best_page} / 유사도 {max_sim:.4f}")
                    csv_writer.writerow([round(result_time, 2), best_page, round(max_sim, 4)])
                    frame_sim_results.append((max_sim, best_page, result_time, gray))

                if frame_sim_results:
                    best_result = max(frame_sim_results, key=lambda x: x[0])
                    prev_sim = best_result[0]
                    last_best_page = best_result[1]
                    last_slide = best_result[3]
                    print(f"📌 현재 상태 갱신 → 페이지 {last_best_page}, 유사도 {prev_sim:.4f}")

    frame_count += 1

cap.release()
csv_file.close()
print("✅ 모든 작업 완료. 로그 저장됨.")
