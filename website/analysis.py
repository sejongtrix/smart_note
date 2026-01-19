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


from io import BytesIO
from torchvision import transforms
import torch.nn.functional as F

# GPU 사용 가능 여부 확인
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper 모델을 전역 변수로 선언 (지연 로드)
whisper_model = None

def pdf2images(pdf_path):
    #pdf_path = r"C:\Users\good1\Documents\카카오톡 받은 파일\제목을-입력해주세요_-2.pdf"
    def extract_images_from_pdf(pdf_path):
        # PDF 파일 이름 (확장자 제거)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 저장할 이미지 리스트
        image_list = []

        # PDF 열기
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_infos = page.get_images(full=True)

            for img_index, img in enumerate(image_infos):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # PIL 이미지 객체 생성
                image = Image.open(BytesIO(image_bytes))

                # 이미지 이름 만들기
                image_name = f"{pdf_name}_{page_num+1}_{img_index+1}.{image_ext}"
                
                # 리스트에 튜플로 저장: (이미지 이름, PIL 이미지 객체)
                image_list.append((image_name, image))

        doc.close()
        return image_list


    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 사전학습 모델 기준
                            std=[0.229, 0.224, 0.225]),
    ])

    def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        return model

    def predict_and_rename_images(image_list, model, transform, device='cuda' if torch.cuda.is_available() else 'cpu'):
        renamed_images = []
        for image_name, image in image_list:
            # RGBA → RGB로 변환
            image = image.convert("RGB")

            # 전처리
            input_tensor = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

            # 추론
            #with torch.no_grad():
            #    outputs = model(input_tensor)
            #    predicted_class = torch.argmax(outputs, dim=1).item()
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                # print("예측 확률:", probs.cpu().numpy())
            # 새로운 이름 지정
            new_image_name = image_name.rsplit('.', 1)[0] + f'_{predicted_class}.' + image_name.rsplit('.', 1)[1]

            renamed_images.append((new_image_name, image))
        return renamed_images


    images = extract_images_from_pdf(pdf_path)
    model = load_model("model_full.pth")

    # 3. 추론 및 이름 변경
    renamed_images= predict_and_rename_images(images, model, transform)
    output_dir = os.path.join("static", "output_images")
    os.makedirs(output_dir, exist_ok=True)
    useful_images = []
    for filename, image in renamed_images:
        save_path = os.path.join(output_dir, filename)
        useful_images.append(filename)
        image.save(save_path)
    print(useful_images)
    del model
    torch.cuda.empty_cache()
    return useful_images
def load_whisper_model():
    """Whisper 모델을 지연 로드합니다."""
    global whisper_model
    if whisper_model is None:
        print("Whisper large 모델을 로드하는 중...")
        whisper_model = whisper.load_model("medium", device=DEVICE)
        print("Whisper 모델 로드 완료.")
    return whisper_model

PROCESSED_FOLDER = 'processed'
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def transcribe_video(video_path, filename,lang):
    """위스퍼를 사용하여 비디오 파일의 음성을 텍스트로 변환합니다."""
    print("받은 lang 값",lang)
    try:
        print(f"{video_path}의 음성 인식을 시작합니다...")
        model = load_whisper_model()
        
        # 더 안정적인 설정으로 음성 인식 수행
        if lang == 0:
            result = model.transcribe(
                video_path, 
                verbose=True, 
                fp16=False,  # fp16을 비활성화하여 안정성 향상
                task='transcribe',  # 번역이 아닌 전사 작업
                temperature=0.2, )
        if lang == 1:
            result = model.transcribe(
                video_path, 
                verbose=True, 
                fp16=False,  # fp16을 비활성화하여 안정성 향상
                task='translate',  # 번역이 아닌 전사 작업
                temperature=0.2,
            )
        
        # 결과 데이터 정리 및 검증
        df = pd.read_csv("slide_similarity_log.csv")
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
        del model
        torch.cuda.empty_cache()
        return transcript_data,segments

    except Exception as e:
        print(f'비디오 처리 중 오류 발생: {str(e)}')
        return {'error': str(e)}

def summarize_ai(raw_df,pdf_path,lang):
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
        if lang == 0:
            prompt = f"""
            문장1의 내용을 바탕으로 문장2를 보완해줘. 단, 문장1에 없는 내용은 추가하지 말고 문장2 안의 표현이나 맥락을 더 구체화하거나 풍부하게 해줘.

            결과는 보완된 문장2만 출력해. 시스템 프롬프트나 설명은 생략하고, 너의 사고과정도 드러내지 마.

            문장1: {text1}
            문장2: {text2}
            """
        if lang == 1:
            prompt = f"""
            The content of sentence 1 is based on the content of sentence 2. Do not add any content that is not in sentence 1, and make the expression or context of sentence 2 more specific or rich.

            Only output the completed sentence 2. Do not include system prompts or explanations, and do not reveal your thought process.
            Sentence1: {text1}
            Sentence2: {text2}

            (((Answer in english.)))
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
            print("text1 : ",text1)
            print("text2 :", text2)
            print(response.text)
            raise Exception("강제로 에러 발생!")
            print(response.text)

    print("ai요약완료")
    output_df = pd.DataFrame(results)
    #output_df.to_csv('gemma_opinions.csv', index=False)
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
        #real_first_page = matching_pdf_and_video(images, video_path)
        matching_pdf_and_video(images,video_path)
        return pdf_data #,real_first_page

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
    import math

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

    # helper: page -> grayscale resized array (for MSE comparisons / last_slide)
    def page_to_gray_resized(page, size=(320, 240)):
        # accepts PIL.Image or numpy array
        if isinstance(page, np.ndarray):
            arr = page
        else:
            arr = np.array(page.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, size)
        return gray

    # helper: ensure image object for feature extraction (PIL.Image)
    def ensure_pil(page):
        if isinstance(page, np.ndarray):
            return Image.fromarray(page).convert("RGB")
        else:
            return page.convert("RGB")

    # 영상 파일 로딩
    print("🎞️ 영상 로딩 중...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"✅ 영상 로딩 완료 - FPS: {fps:.2f}, Frame count: {frame_total}")
    frame_interval = max(1, int(fps * 1))

    def is_last_page(page_index, doc_length):
        return page_index == doc_length - 1

    # 상태 변수들
    last_slide = None
    #last_best_page = 0
    prev_sim = None
    frame_count = 0

    # 하이퍼파라미터
    mse_threshold = 500
    sim_drop_threshold = 0.01
    max_search_range = 20

    # ---- 새로 추가된 부분: 초기 첫 슬라이드 후보 결정 ----
    # 비디오 초반 프레임 몇 개를 추출해서, 문서의 처음 up_to_n_pages 중에서 평균 유사도 최대인 페이지를 첫 슬라이드로 선택
    def choose_initial_slide(num_pages_to_check=5, sample_frame_count=3, sample_span_seconds=1.0):
        nonlocal last_best_page, last_slide

        if len(images) == 0:
            return

        # 실제로 체크할 페이지 수
        up_to = min(num_pages_to_check, len(images))
        candidate_indices = list(range(up_to))
        print(f"🔰 초기 슬라이드 후보 인덱스: {candidate_indices}")

        # 샘플 프레임 시간 간격 (seconds)
        span = sample_span_seconds
        # sample_frame_count 프레임을 뽑되, 영상 전체에 걸쳐 적절 간격으로 뽑음 (초반부만 사용하려면 start at 0..)
        # 여기서는 영상 시작 ~ (span * sample_frame_count) 영역 내에서 frame_interval 간격으로 샘플
        sampled_features = []
        sampled_times = []

        # 안전: 영상 프레임이 적으면 가능한 프레임들만 사용
        for s_idx in range(sample_frame_count):
            # 시간 위치 (s_idx * span)
            t = s_idx * span
            frame_pos = int(min(frame_total - 1, max(0, round(t * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, f = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            feat = extract_feature(rgb)
            sampled_features.append(feat)
            sampled_times.append(frame_pos / fps)

        if not sampled_features:
            print("⚠️ 초기 샘플 프레임을 읽어오지 못했습니다. 기본(0)으로 설정합니다.")
            last_best_page = 0
            last_slide = page_to_gray_resized(images[0])
            return

        # 각 후보 페이지에 대해 sampled_features와의 평균 유사도 계산
        avg_sims = []
        for idx in candidate_indices:
            page = images[idx]
            pil_page = ensure_pil(page)
            feat_page = extract_feature(pil_page)
            sims = []
            for sf in sampled_features:
                sim = cosine_similarity(sf.unsqueeze(0), feat_page.unsqueeze(0)).item()
                sims.append(sim)
            avg = float(np.mean(sims))
            avg_sims.append((idx, avg))
            print(f"    후보 페이지 {idx} 평균 유사도: {avg:.4f}")

        # 최고 평균 유사도 페이지 선택
        best_idx, best_avg = max(avg_sims, key=lambda x: x[1])
        last_best_page = best_idx
        last_slide = page_to_gray_resized(images[best_idx])
        print(f"🏁 초기 선택 완료 → 페이지 {best_idx} (평균 유사도 {best_avg:.4f})")
        # 리턴해서 디버그에 사용 가능
        return best_idx, best_avg, sampled_times

    print("🚀 분석 시작 (초기 슬라이드 선택 중...)")
    # 선택 수행 (여기서 필요하면 파라미터를 바꿀 수 있음)
    last_best_page,trash1,trash2 = choose_initial_slide(num_pages_to_check=5, sample_frame_count=3, sample_span_seconds=0.5)
    real_first_page = last_best_page
    # 초기 샘플링 때문에 비디오 포지션이 움직였으므로 루프 시작 전 프레임 포지션을 0으로 돌림
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

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
                print("🆕 (루프 중) 첫 슬라이드로 설정")
                last_slide = gray
            else:
                diff = mse(gray, last_slide)
                #print(f"🔍 MSE 차이: {diff:.2f}")

                if diff > mse_threshold:
                    print("📈 슬라이드 변경 감지됨 → 탐색 시작")
                    frame_sim_results = []

                    for offset in range(-int(2 * fps), int(2 * fps) + 3, frame_interval):
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
                            feat2 = extract_feature(ensure_pil(page))
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
                                feat2 = extract_feature(ensure_pil(page))
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
    del model
    torch.cuda.empty_cache()
    print("✅ 모든 작업 완료. 로그 저장됨.")
    return real_first_page