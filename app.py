from flask import Flask, render_template, request, jsonify, send_file, url_for, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
from analysis import transcribe_video, process_pdf, summarize_ai
import ffmpeg
import pandas as pd
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

@app.route('/result')
def result():
    # 처리된 파일들 찾기
    transcript_files = []
    pdf_files = []
    
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
        name,ext = os.path.splitext(video_file_name)
        transcript_filename = f"{name}_transcript.json"
        transcript_path = os.path.join(PROCESSED_FOLDER, transcript_filename)

        # 해당 파일이 존재할 때만 열기
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        else:
            print(f"Transcript file not found: {transcript_path}")
            transcript_data = None
    
    if pdf_files:
        latest_pdf = max(pdf_files, key=lambda x: os.path.getctime(os.path.join(PROCESSED_FOLDER, x)))
        with open(os.path.join(PROCESSED_FOLDER, latest_pdf), 'r', encoding='utf-8') as f:
            #global pdf_path_ai
            #print(pdf_path_ai)
            #pdf_path_ai = os.path.join(PROCESSED_FOLDER, latest_pdf)
            pdf_data = json.load(f)
    global last_uploaded_audio_path
    #print(latest_pdf)
    audio_filename = os.path.basename(last_uploaded_audio_path)
    ai_data = summarize_ai(segments_raw,pdf_path_ai) # 여기서 요약 들어간다
    return render_template('result.html', transcript_data=transcript_data, pdf_data=pdf_data, audio_filename=audio_filename, ai_data = ai_data.to_json(orient='records',force_ascii=False))


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
        try:
            transcript_data,segments_raw = transcribe_video(filepath, filename)
            global last_uploaded_audio_path
            last_uploaded_audio_path = save_audio(filepath)
            if 'error' in transcript_data:
                return jsonify({'error': f'비디오 처리 중 오류: {transcript_data["error"]}'}), 500

            return jsonify({
                'success': True,
                'message': '비디오 음성 인식이 완료되었습니다.',
                'transcript': transcript_data['transcript'],
                'segments': transcript_data['segments']
            })
            
        except Exception as e:
            return jsonify({'error': f'비디오 처리 중 오류가 발생했습니다: {str(e)}'}), 500
    
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

    df = pd.read_csv("C:/Users/good1/Desktop/summer_vacation/smartnote/website/notes/slide_similarity_log.csv")
    
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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
