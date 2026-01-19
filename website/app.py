from flask import Flask, render_template, request, jsonify, send_file, url_for, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
from analysis import transcribe_video, process_pdf, summarize_ai, pdf2images
import ffmpeg
import pandas as pd
import base64
import time
import requests
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 허용된 파일 확장자
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_PDF_EXTENSIONS = {'pdf'}
# mp3 변환#
def save_audio(video_path):
    base, _ = os.path.splitext(video_path)
    audio_output_path = base + ".mp3"
    ffmpeg.input(video_path).output(audio_output_path).run(overwrite_output=True)
    return audio_output_path
# mp3 변환#
def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index_eng')
def index_eng():
    return render_template('index_eng.html')

@app.route('/final_refine', methods=['GET'])
def final_refine():
    # 처리된 파일들 찾기
    transcript_files = []
    pdf_files = []
    no_use_transcript, segments_raw = transcribe_video(last_uploaded_video_path, video_file_name,0)
    
    if os.path.exists(PROCESSED_FOLDER):
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.endswith('_transcript.json'):
                transcript_files.append(filename)
            elif filename.endswith('_processed.json'):
                pdf_files.append(filename)
    
    # 가장 최근 파일들 읽기
    transcript_data = None
    pdf_data = None
    if transcript_files:
        name, ext = os.path.splitext(video_file_name)
        transcript_filename = f"{name}_transcript.json"
        transcript_path = os.path.join(PROCESSED_FOLDER, transcript_filename)

        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        else:
            print(f"Transcript file not found: {transcript_path}")
    
    if pdf_files:
        latest_pdf = max(pdf_files, key=lambda x: os.path.getctime(os.path.join(PROCESSED_FOLDER, x)))
        with open(os.path.join(PROCESSED_FOLDER, latest_pdf), 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
    
    global last_uploaded_audio_path
    audio_filename = os.path.basename(last_uploaded_audio_path)
    global ai_data
    ai_data = summarize_ai(segments_raw, pdf_path_ai,0)
    image_files = pdf2images(pdf_path_ai)
    global transcript_data2
    transcript_data2 = transcript_data
    global pdf_data2
    pdf_data2 = pdf_data
    global audio_filename2
    audio_filename2 = audio_filename
    global image_files2
    image_files2 = image_files
    # Check if the request is from fetch
    return jsonify({'success': True, 'message': 'Processing complete.'})
    
    # Render the result page for normal requests
    #return render_template(
    #    'result.html',
    #    transcript_data=transcript_data,
    #    pdf_data=pdf_data,
    #    audio_filename=audio_filename,
    #    ai_data=ai_data.to_json(orient='records', force_ascii=False),
    #    image_files=image_files
    #)

@app.route('/final_refine_eng', methods=['GET'])
def final_refine_eng():
    # 처리된 파일들 찾기
    transcript_files = []
    pdf_files = []
    no_use_transcript, segments_raw = transcribe_video(last_uploaded_video_path, video_file_name,1)
    
    if os.path.exists(PROCESSED_FOLDER):
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.endswith('_transcript.json'):
                transcript_files.append(filename)
            elif filename.endswith('_processed.json'):
                pdf_files.append(filename)
    
    # 가장 최근 파일들 읽기
    transcript_data = None
    pdf_data = None
    if transcript_files:
        name, ext = os.path.splitext(video_file_name)
        transcript_filename = f"{name}_transcript.json"
        transcript_path = os.path.join(PROCESSED_FOLDER, transcript_filename)

        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        else:
            print(f"Transcript file not found: {transcript_path}")
    
    if pdf_files:
        latest_pdf = max(pdf_files, key=lambda x: os.path.getctime(os.path.join(PROCESSED_FOLDER, x)))
        with open(os.path.join(PROCESSED_FOLDER, latest_pdf), 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
    
    global last_uploaded_audio_path
    audio_filename = os.path.basename(last_uploaded_audio_path)
    global ai_data
    ai_data = summarize_ai(segments_raw, pdf_path_ai,1)
    image_files = pdf2images(pdf_path_ai)
    global transcript_data2
    transcript_data2 = transcript_data
    global pdf_data2
    pdf_data2 = pdf_data
    global audio_filename2
    audio_filename2 = audio_filename
    global image_files2
    image_files2 = image_files
    # Check if the request is from fetch
    return jsonify({'success': True, 'message': 'Processing complete.'})
    
    # Render the result page for normal requests
    #return render_template(
    #    'result.html',
    #    transcript_data=transcript_data,
    #    pdf_data=pdf_data,
    #    audio_filename=audio_filename,
    #    ai_data=ai_data.to_json(orient='records', force_ascii=False),
    #    image_files=image_files
    #)



@app.route('/result', methods=['GET'])
def result():
    return render_template(
        'result.html',
        transcript_data=transcript_data2,
        pdf_data=pdf_data2,
        audio_filename=audio_filename2,
        ai_data=ai_data.to_json(orient='records', force_ascii=False),
        image_files=image_files2
    )

@app.route('/result_eng', methods=['GET'])
def result_eng():
    return render_template(
        'result_eng.html',
        transcript_data=transcript_data2,
        pdf_data=pdf_data2,
        audio_filename=audio_filename2,
        ai_data=ai_data.to_json(orient='records', force_ascii=False),
        image_files=image_files2
    )
@app.route('/output_images/<path:filename>')
def output_images(filename):
    # 보안상 safe join/check 추가 가능
    full = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/get_opinion')
def get_opinion():
    page = int(request.args.get('page', 0))
    gemma_opinion = ai_data.loc[ai_data['page'] == page, 'gemma_response'].values[0]
    #print(gemma_opinion)
    return jsonify(response=gemma_opinion)

@app.route('/uploads/<filename>') # mp3파일 재생하려고 만듦
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<path:filename>')  # PDF 파일 불러오기
def serve_processed_file(filename):
    # 전역변수에서 PDF 파일 이름 가져오기
    global fffilename
    print(fffilename)
    if fffilename:
        # PDF 파일 이름 추출
        pdf_filename = os.path.splitext(os.path.basename(fffilename))[0] + "_page_" + filename
        pdf_path = os.path.join(PROCESSED_FOLDER, pdf_filename)
        #pdf_path = pdf_filename
        print(pdf_path,"을 반환했어")
        global pdf_path_ai
        pdf_path_ai =pdf_path
        # 파일이 존재하는지 확인
        if os.path.exists(pdf_path):
            return send_file(pdf_path)
        else:
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
    else:
        return jsonify({'error': 'PDF 파일 이름이 설정되지 않았습니다.'}), 400

#@app.route('/upload_ai',methods=['POST'])
#def upload_ai():
#    return jsonify({
#        'success': True,
#        'message': 'ai 요약이 완료되었습니다.'
#    })


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        global last_uploaded_video_path
        global video_file_name
        global segments_raw
        video_file_name = filename
        last_uploaded_video_path = filepath
        #try:
        global last_uploaded_audio_path
        last_uploaded_audio_path = save_audio(filepath)
        #    transcript_data,segments_raw = transcribe_video(filepath, filename)
        #    if 'error' in transcript_data:
        #        return jsonify({'error': f'비디오 처리 중 오류: {transcript_data["error"]}'}), 500

        return jsonify({
            'success': True,
            'message': '비디오 업로드가 완료되었습니다.',
            #'transcript': transcript_data['transcript'],
            #'segments': transcript_data['segments']
        })
            
        #except Exception as e:
        #    return jsonify({'error': f'비디오 처리 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'PDF 파일이 선택되지 않았습니다.'}), 400
    global last_uploaded_video_path
    if last_uploaded_video_path is None:
        return jsonify({"동영상 먼저 올려주세요."}),400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        global pdf_path_ai
        pdf_path_ai = filepath
        global fffilename
        fffilename = filename
        try:
            pdf_data = process_pdf(filepath, filename,last_uploaded_video_path)
            #global initial_first_page
            #initial_first_page= real_first_page
            if 'error' in pdf_data:
                 return jsonify({'error': f'PDF 처리 중 오류: {pdf_data["error"]}'}), 500
            return jsonify({
                'success': True,
                'message': 'PDF 처리가 완료되었습니다.',
                'text_content': pdf_data['text_content'][:5000] + "..." if len(pdf_data['text_content']) > 5000 else pdf_data['text_content'],
                'total_pages': pdf_data['total_pages'],
                'image_paths': pdf_data['image_paths']
            })
            
        except Exception as e:
            return jsonify({'error': f'PDF 처리 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400


#@app.route('/processed/<filename>')
#def get_processed_image(filename):
#    try:
#        return send_file(os.path.join(PROCESSED_FOLDER, filename))
#    except FileNotFoundError:
#        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404

@app.route('/summarize', methods=['POST'])
def summarize_content():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '요약할 텍스트가 없습니다.'}), 400
        
        # 임시 모의 요약 결과
        final_summary = f"""
        텍스트 요약 결과 (임시 모의 데이터):

        주요 내용:
        - 인공지능의 기초 개념과 원리
        - 머신러닝과 딥러닝의 차이점
        - 실제 산업 응용 사례들
        - AI 기술의 미래 발전 방향

        핵심 포인트:
        1. 데이터의 중요성과 전처리 과정
        2. 다양한 알고리즘의 선택 기준
        3. 모델 평가와 최적화 방법
        4. 윤리적 AI 개발의 필요성

        결론:
        인공지능은 현대 사회에 혁신적인 변화를 가져오고 있으며, 지속적인 연구와 개발이 필요합니다.

        (실제 LLM 모델 설치 후 실제 요약 기능이 동작합니다)
        """
        
        return jsonify({
            'success': True,
            'summary': final_summary
        })
        
    except Exception as e:
        return jsonify({'error': f'요약 처리 중 오류가 발생했습니다: {str(e)}'}), 500
    

@app.route("/current_page", methods=["POST"])
def current_page():
    data = request.get_json()
    current_time = data.get("current_time", 0)  # 클라이언트에서 보낸 현재 재생 시간

    df = pd.read_csv("slide_similarity_log.csv")
    
    # page가 바뀌는 지점만 추출
    segments = []
    prev_page = None
    for i, row in df.iterrows():
        if row['page'] != prev_page:
            segments.append({"time": row['seconds'], "page": int(row['page'])})
            prev_page = row['page']
    
    # 현재 시간에 해당하는 페이지 찾기
    current_page = 0
    for segment in segments:
        if current_time >= segment["time"]:
            current_page = segment["page"]
        else:
            break

    return jsonify({"current_page": current_page})

################## 여긴 해림님의 코드 ############################
import mimetypes
def image_to_data_uri(path):
    print(path)
    relative_path = path.split("://", 1)[-1]  # '192.168.0.44:5001/static/output_images/test33_1_1_3.jpeg'
    relative_path2 = relative_path.split("/", 1)[-1] 
    #relative_path = path.split("/static/", 1)[-1]
    print(relative_path2)
    with open(relative_path2, "rb") as f:
        mime_type, _ = mimetypes.guess_type(relative_path2)
        encoded = base64.b64encode(f.read()).decode("utf-8")
        print("uri성공",mime_type)
    return f"data:{mime_type};base64,{encoded}"

#API_KEY = "msy_dummy_api_key_for_test_mode_12345678"  # Test API key
API_KEY = "msy_x7YOddddBotIhwDmRKtK7GgUQ5JCE15DOoGA"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
BASE = "https://api.meshy.ai/openapi/v1"

def create_task(image_url):  # Image to 3D
    payload = {
        "image_url": image_url,
        "enable_pbr": False,
        "should_remesh": True,
        "should_texture": True
    }
    resp = requests.post(f"{BASE}/image-to-3d", json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["result"]

def get_task_status(task_id):  # Check task status
    resp = requests.get(f"{BASE}/image-to-3d/{task_id}", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def download_fbx_from_url(fbx_url, task_id, save_dir="3D"):
    if not fbx_url:
        print("fbx URL이 없습니다.")
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_id}.fbx")

    response = requests.get(fbx_url)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    print(f"fbx 저장 완료: {save_path}")
    return save_path

# Flask route for 3D model generation
@app.route('/generate_3d', methods=['POST'])
def generate_3d_model():
    try:
        # JSON 데이터에서 파일 이름 가져오기
        data = request.get_json()
        filename = data.get('filename', '').strip()
        print(filename,"현재 filename은")
        if not filename:
            return jsonify({'error': '파일 이름이 제공되지 않았습니다.'}), 400

        # 파일 경로 생성
        print("-1")
        filepath = os.path.join(filename)
        #if not os.path.exists(filepath):
        #    return jsonify({'error': '파일이 존재하지 않습니다.'}), 404
        print("0")
        # Convert image to data URI and create 3D model task
        img_url = image_to_data_uri(filepath)
        print("1")
        task_id = create_task(img_url)
        print("2")
        print("Created task:", task_id)

        # Poll for task status
        while True:
            status_info = get_task_status(task_id)
            status = status_info.get("status")
            progress = status_info.get("progress", 0)

            print(f"현재 상태: {status} | 진행률: {progress}%")

            if status == "SUCCEEDED":
                model_urls = status_info.get("model_urls", {})
                fbx_url = model_urls.get("fbx")
                print("FBX URL:", fbx_url)
                break
            elif status == "FAILED":
                return jsonify({'error': '3D 모델 생성 작업이 실패했습니다.'}), 500
            time.sleep(1)

        # Download the FBX file
        fbx_path = download_fbx_from_url(fbx_url, task_id)

        # Use Selenium to upload the FBX file to the viewer
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        driver.get("https://imagetostl.com/kr/view-fbx-online")

        time.sleep(1)  # Wait for the page to load

        file_input = driver.find_element(By.XPATH, '//input[@type="file"]')
        file_input.send_keys(os.path.abspath(fbx_path))

        # Wait for the user to close the browser
        try:
            while True:
                if not driver.window_handles:
                    print("브라우저 창이 모두 닫혀서 스크립트 종료합니다.")
                    break
                time.sleep(1)
        except WebDriverException:
            print("브라우저가 닫혔습니다. 스크립트 종료합니다.")
        finally:
            driver.quit()

        return jsonify({'success': True, 'message': '3D 모델 생성 및 FBX 파일 처리가 완료되었습니다.'})

    except Exception as e:
        return jsonify({'error': f'3D 모델 생성 중 오류가 발생했습니다: {str(e)}'}), 500


###########해림님의 코드







if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
