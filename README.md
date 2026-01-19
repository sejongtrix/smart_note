# Video + PDF Lecture Analyzer (Flask) + Slide–Video Matching + Image→3D (Meshy)

강의 **동영상(mp4 등)** 과 **PDF 자료**를 업로드하면,

* Whisper로 **음성 전사/번역(ko→en)** 을 수행하고
* PDF를 **페이지 이미지 + 텍스트**로 처리한 뒤
* 동영상 프레임과 PDF 슬라이드를 **유사도 기반으로 매칭(페이지 추적)**
* 페이지 단위로 **(동영상 전사 텍스트 + PDF 텍스트) → LLM(Gemma3:12B/Ollama) 보완 요약**
* PDF 내 이미지들을 추출해 **분류 모델로 라벨링 후 static 폴더에 저장**
* (옵션) 이미지 기반 **3D 모델(FBX) 생성(Meshy API)** 및 뷰어 업로드(Selenium)

까지 한 번에 처리하는 Flask 웹 앱입니다.

---

## 1) 주요 기능

### A. 업로드 & 처리 플로우

1. `/upload_video`

   * 동영상 업로드 → `ffmpeg` 로 **mp3 추출**
2. `/upload_pdf`

   * PDF 업로드 → 텍스트 추출(PyPDF2) + 페이지 이미지 변환(pdf2image)
   * `matching_pdf_and_video()` 실행 → **slide_similarity_log.csv** 생성
3. `/final_refine` 또는 `/final_refine_eng`

   * Whisper 전사/번역 + PDF 텍스트 기반 보완 요약(LLM)
   * PDF 내 이미지 추출 + 분류 모델로 파일명 라벨링
4. `/result` 또는 `/result_eng`

   * 처리 결과 페이지 렌더링

### B. 슬라이드-비디오 매칭 (핵심)

* ResNet50 feature + cosine similarity 기반으로 프레임과 PDF 페이지를 매칭합니다.
* MSE(픽셀 차)로 **슬라이드 변경 감지** → 변경 시점에서 후보 페이지 탐색
* 결과는 `slide_similarity_log.csv`로 저장합니다.

### C. 페이지별 LLM 보완 요약

* 동영상 전사 텍스트(페이지별로 그룹) + 해당 PDF 페이지 텍스트를 함께 사용하여,
* **문장1(전사 기반)** 을 기준으로 **문장2(PDF 텍스트)** 를 “추가 내용 없이” 더 구체화합니다.
* Ollama 서버에서 `gemma3:12b`로 요청합니다.

### D. PDF 내 이미지 추출 및 분류

* `fitz(PyMuPDF)`로 PDF 내 이미지를 추출하고
* `model_full.pth`로 예측하여 파일명에 `_class`를 붙여 저장합니다.
* 결과 이미지는 `static/output_images/`에 저장됩니다.

### E. (옵션) 이미지 → 3D (Meshy API)

* 입력 이미지(Data URI)를 Meshy Image-to-3D로 전송하여 FBX 생성
* 생성된 FBX를 다운로드 후 Selenium으로 웹 뷰어에 업로드합니다.

---

## 2) 프로젝트 구조(예시)

```bash
project/
├─ app.py                     # Flask 서버 (업로드/결과/3D 등)
├─ analysis.py                 # 전사/요약/PDF처리/슬라이드매칭/이미지추출
├─ templates/
│  ├─ index.html
│  ├─ index_eng.html
│  ├─ result.html
│  └─ result_eng.html
├─ static/
│  └─ output_images/           # PDF에서 추출한 이미지 저장
├─ uploads/                    # 업로드 파일 저장
├─ processed/                  # PDF 페이지 이미지 + json 산출물
├─ slide_similarity_log.csv    # 슬라이드-비디오 매칭 로그
└─ model_full.pth              # PDF 이미지 분류 모델
```

---

## 3) 실행 방법

### 3.1 필수 설치 (Python)

권장 requirements (환경에 맞게 조정):

* flask, werkzeug
* pandas
* requests
* ffmpeg-python
* openai-whisper
* torch, torchvision
* PyMuPDF(fitz), PyPDF2, pdf2image, pillow
* selenium, webdriver-manager
* opencv-python
* poppler

### 3.2 Ollama 실행 (LLM 요약)

```bash
ollama serve
ollama pull gemma3:12b
```

* 기본 요청 URL: `http://localhost:11434/api/generate`

### 3.3 서버 실행

```bash
python app.py
```

* 기본: `http://0.0.0.0:5001`

---

## 4) 사용 흐름 (UI 기준)

1. `/` 접속 (KOR) 또는 `/index_eng` (ENG)
2. 동영상 업로드 → PDF 업로드
3. `/final_refine` 또는 `/final_refine_eng`
4. `/result` 또는 `/result_eng`

---


## 5) 핵심 로직 설명

### 5.1 슬라이드 매칭 로그(slide_similarity_log.csv)

* 컬럼: `seconds, page, similarity`
* 프레임을 일정 간격으로 읽고(Min 1초 간격)
* 슬라이드가 바뀐 것으로 감지되면 후보 페이지를 탐색하여
* “이 시간대에서 가장 그럴듯한 페이지”를 CSV에 기록합니다.

### 5.2 전사 결과에 페이지 붙이기

* Whisper segment의 `start time`을 기준으로
* `slide_similarity_log.csv`에서 “그 시점의 페이지”를 매핑하여
* segment마다 `page` 필드를 붙입니다.

### 5.3 LLM 보완 요약(summarize_ai)

* page별로 전사 text를 합쳐 `text1`으로 만들고,
* PDF page 텍스트를 `text2`로 가져온 뒤
* `text1에 없는 내용은 추가하지 말고 text2를 더 구체화` 프롬프트로 Ollama 호출합니다.

---
