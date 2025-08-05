import os
import json
import whisper
import torch
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import pandas as pd
import requests
import fitz
# GPU 사용 가능 여부 확인
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper 모델을 전역 변수로 선언 (지연 로드)
whisper_model = None

def load_whisper_model():
    """Whisper 모델을 지연 로드합니다."""
    global whisper_model
    if whisper_model is None:
        print("Whisper medium 모델을 로드하는 중...")
        whisper_model = whisper.load_model("tiny", device=DEVICE)
        print("Whisper 모델 로드 완료.")
    return whisper_model

PROCESSED_FOLDER = 'processed'
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def transcribe_video(video_path, filename):
    """위스퍼를 사용하여 비디오 파일의 음성을 텍스트로 변환합니다."""
    try:
        print(f"{video_path}의 음성 인식을 시작합니다...")
        model = load_whisper_model()
        
        # 더 안정적인 설정으로 음성 인식 수행
        result = model.transcribe(
            video_path, 
            verbose=True, 
            fp16=False,  # fp16을 비활성화하여 안정성 향상
            task='transcribe',  # 번역이 아닌 전사 작업
            temperature=0.2,  # 일관된 결과를 위해 temperature를 0으로 설정
            #no_speech_threshold=0.6,  # 음성이 없는 구간 감지 임계값
            #logprob_threshold=-1.0,  # 로그 확률 임계값
            #compression_ratio_threshold=2.4  # 압축 비율 임계값
        )
        
        # 결과 데이터 정리 및 검증
        df = pd.read_csv("C:/Users/good1/Desktop/summer_vacation/smartnote/website/notes/slide_similarity_log.csv")
        page_segments = []
        prev_page = None
        for i, row in df.iterrows():
            if row['page'] != prev_page:
                page_segments.append({"time": row['seconds'], "page": int(row['page'])})
                prev_page = row['page']
        segments = []
        for segment in result.get('segments', []):
            segment_start = segment.get('start',0.0)
            current_page = 0
            for pseg in page_segments:
                if segment_start >= pseg['time']:
                    current_page = pseg['page']
                else:
                    break
            segments.append({
                'start': segment.get('start', 0.0),
                'end': segment.get('end', 0.0),
                'text': segment.get('text', '').strip(),
                'page': current_page
            })
        
        transcript_data = {
            'filename': filename,
            'transcript': result.get('text', '').strip(),
            'segments': segments,
            'language': result.get('language', 'unknown')
        }
        transcript_file = os.path.join(PROCESSED_FOLDER, f"{os.path.splitext(filename)[0]}_transcript.json")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
        print("음성 인식 완료 및 파일 저장 성공.")
        return transcript_data,segments

    except Exception as e:
        print(f'비디오 처리 중 오류 발생: {str(e)}')
        return {'error': str(e)}

def summarize_ai(raw_df,pdf_path):
    df = pd.DataFrame(raw_df)
    grouped = df.groupby('page', as_index=False).agg({
    'text': ' '.join  
    })

    example_df = grouped

    doc = fitz.open(pdf_path)

    # 페이지별 텍스트 저장
    data = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        data.append({
            'page': page_num,
            'text': text.strip()
        })

    # DataFrame으로 저장
    page_df = pd.DataFrame(data)


    # Ollama 서버 URL
    OLLAMA_URL = 'http://localhost:11434/api/generate'

    # 결과 저장용 리스트
    results = []

    for idx, row in example_df.iterrows():
        text1 = row['text'] 

        page_num = row['page']  # page 번호
        # page_df에서 해당하는 page의 text 가져오기
        matched_rows = page_df[page_df['page'] == page_num]

        if matched_rows.empty:
            print(f"❗ page {page_num}에 해당하는 텍스트가 없습니다.")
            continue

        text2 = matched_rows.iloc[0]['text']

        # 프롬프트 구성
        prompt = f"""
        문장1의 내용을 바탕으로 문장2를 보완해줘. 단, 문장1에 없는 내용은 추가하지 말고 문장2 안의 표현이나 맥락을 더 구체화하거나 풍부하게 해줘.

        결과는 보완된 문장2만 출력해. 시스템 프롬프트나 설명은 생략하고, 너의 사고과정도 드러내지 마.

        문장1: {text1}
        문장2: {text2}
        """

        # 요청 payload
        payload = {
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False
        }

        # 요청 보내기
        response = requests.post(OLLAMA_URL, json=payload)
        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            print(f"(index {idx}):")
            print(result['response'])
            results.append({
                'index': idx,
                'page': page_num,
                'text1': text1,
                'text2': text2,
                'gemma_response': result['response']
            })
        else:
            print(f"❌ 요청 실패 (index {idx}): {response.status_code}")
            print(response.text)

    print("ai요약완료")
    output_df = pd.DataFrame(results)
    output_df.to_csv('gemma_opinions.csv', index=False)
    output_df.to_json('gemma_opinions.json', orient='records', force_ascii=False, indent=2)
    return output_df

def process_pdf(pdf_path, filename,video_path):
    """PDF 파일을 처리하여 텍스트와 이미지로 변환합니다."""
    # 여기서 pdf 처리 로직 다 구현 해버리자고 
    try:
        print(f"{pdf_path}의 PDF 처리를 시작합니다...")
        # 1. PDF에서 텍스트 추출
        text_content = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_content += page.extract_text() + "\n\n"

        # 2. PDF를 이미지로 변환
        images = convert_from_path(pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_filename = f"{os.path.splitext(filename)[0]}_page_{i+1}.png"
            image_filepath = os.path.join(PROCESSED_FOLDER, image_filename)
            image.save(image_filepath, 'PNG')
            image_paths.append(image_filename)

        pdf_data = {
            'filename': filename,
            'text_content': text_content,
            'image_paths': image_paths,
            'total_pages': len(images)
        }

        pdf_file = os.path.join(PROCESSED_FOLDER, f"{os.path.splitext(filename)[0]}_processed.json")
        with open(pdf_file, 'w', encoding='utf-8') as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=2)

        print("PDF 처리 및 파일 저장 성공.")
        # 3. pdf와 동영상 파일로 
        matching_pdf_and_video(images, video_path)
        return pdf_data

    except Exception as e:
        print(f'PDF 처리 중 오류 발생: {str(e)}')
        return {'error': str(e)}
    
def matching_pdf_and_video(images, video_path):
    """PDF 데이터와 비디오 데이터를 매칭합니다."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    from torch.nn.functional import cosine_similarity
    import io
    import cv2
    import numpy as np
    import csv

    print("🔧 모델 로딩 중...")
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1]) 
    model.eval()
    print("✅ 모델 로딩 완료")


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
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 영상 로딩 완료 - FPS: {fps:.2f}")
    frame_interval = int(fps * 1)
    def is_last_page(page_index,doc_length):
        return page_index == doc_length - 1
    # 상태 변수들
    last_slide = None
    last_best_page = 0
    prev_sim = None
    frame_count = 0

    # 하이퍼파라미터
    mse_threshold = 500
    sim_drop_threshold = 0.01
    max_search_range = 20

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
                        candidates = [last_best_page + i for i in candidate_range if 0 <= last_best_page + i < len(images)]

                        best_page = last_best_page
                        max_sim = -1

                        print(f"🔎 1차 탐색: 후보 페이지 {candidates}")
                        for i in candidates:
                            page = images[i]
                            #pix = page.get_pixmap(dpi=300)
                            #img_bytes = pix.tobytes("ppm")
                            feat2 = extract_feature(page)
                            sim = cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

                            print(f"    페이지 {i} 유사도: {sim:.4f}")
                            if sim > max_sim:
                                max_sim = sim
                                best_page = i
                                best_feat2 = feat2

                        # 유사도 급락 시 재탐색
                        if prev_sim is not None and (prev_sim - max_sim) >= sim_drop_threshold:
                            print(f"⚠️ 유사도 하락 감지 (이전: {prev_sim:.4f} → 현재: {max_sim:.4f}) → 2차 탐색")
                            expanded_range = list(range(-(max_search_range//2), max_search_range//2 + 1))
                            expanded_candidates = [last_best_page + i for i in expanded_range if 0 <= last_best_page + i < len(images)]

                            print(f"🔍 2차 탐색: 후보 페이지 {expanded_candidates}")
                            for i in expanded_candidates:
                                page = images[i]
                                #pix = page.get_pixmap(dpi=300)
                                #img_bytes = pix.tobytes("ppm")
                                feat2 = extract_feature(page)
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
                        
                        if is_last_page(last_best_page, len(images)):
                            print(f"🚩 마지막 페이지({last_best_page}) 도달, 종료합니다.")
                            break

        frame_count += 1

    cap.release()
    csv_file.close()
    print("✅ 모든 작업 완료. 로그 저장됨.")
