// 전역 변수
let uploadedVideos = [];
let uploadedPDFs = [];
let transcriptData = null;
let pdfData = null;
let isVideoUploaded = false;
let isPDFUploaded = false;
let readyisready = false;
// 최대 파일 수 제한
const MAX_VIDEOS = 1;
const MAX_PDFS = 1;

// DOM 로드 완료 후 실행
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateRemoveButtonsVisibility();
    updateAddButtonState();
});

function checkAllUploadsComplete() {
    if (isVideoUploaded && isPDFUploaded) {
        updateProgress(80);
        showNotification('모든 파일이 업로드완료되었습니다. 후처리 페이지로 이동합니다.', 'success');
        showLoading('각종 후 처리 하는중');

        // Redirect to /result and wait for it to complete
        fetch('/final_refine')
            .then(response => {
                if (response.ok) {
                    return response.json(); // 서버 응답 대기
                } else {
                    throw new Error('Result processing failed.');
                }
            })
            .then(() => {
                // 요청이 성공적으로 완료되면 후처리 완료 알림 및 페이지 이동
                turntoresult();
            })
            .catch(error => {
                console.error(error);
                showNotification('결과 처리 중 오류가 발생했습니다.', 'error');
            });
    }
}

function turntoresult() {
    updateProgress(100);
    showNotification('모든 후처리가 완료되었습니다', 'success');
    setTimeout(() => {
        window.location.href = '/result';  // Flask에서 이 라우트를 처리해야 함
    }, 1000); // 사용자에게 메시지를 보여줄 시간
}
// 이벤트 리스너 초기화
function initializeEventListeners() {
    // 초기 파일 업로드 이벤트
    setupFileUploadEvents();
    
    // 탭 및 요약 버튼이 삭제되어 이벤트 리스너 제거
}

// 파일 업로드 이벤트 설정
function setupFileUploadEvents() {
    // 모든 비디오 입력에 이벤트 리스너 추가
    document.querySelectorAll('.video-input').forEach(input => {
        input.addEventListener('change', handleVideoUpload);
    });
    
    // 모든 PDF 입력에 이벤트 리스너 추가
    document.querySelectorAll('.pdf-input').forEach(input => {
        input.addEventListener('change', handlePDFUpload);
    });
}

// 비디오 업로드 처리
function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const input = event.target;
    const uploadBtn = input.nextElementSibling;
    const removeBtn = uploadBtn.nextElementSibling;
    
    uploadBtn.textContent = file.name;
    uploadBtn.classList.add('file-selected');
    removeBtn.style.display = 'flex';
    
    uploadedVideos.push(file);
    updateProgress(0);
    updateRemoveButtonsVisibility();
    
    // 서버에 업로드
    uploadVideoToServer(file);
}

// PDF 업로드 처리
function handlePDFUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const input = event.target;
    const uploadBtn = input.nextElementSibling;
    const removeBtn = uploadBtn.nextElementSibling;
    
    uploadBtn.textContent = file.name;
    uploadBtn.classList.add('file-selected');
    removeBtn.style.display = 'flex';
    
    uploadedPDFs.push(file);
    updateProgress(30);
    updateRemoveButtonsVisibility();
    
    // 서버에 업로드
    uploadPDFToServer(file);
}

// 비디오 서버 업로드
async function uploadVideoToServer(file) {
    showLoading('비디오를 업로드하고 음성을 인식하는 중...');
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            //transcriptData = result;
            updateProgress(10);
            showNotification('비디오 업로드 완료되었습니다.', 'success');
            isVideoUploaded = true; 
            checkAllUploadsComplete(); 
        } else {
            showError('비디오 업로드 실패: ' + result.error);
        }
    } catch (error) {
        showError('비디오 업로드 중 오류가 발생했습니다: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function transfering_ai(file) {
    showLoading('각종 후 처리 하는중');
    
    const formData = new FormData();
    formData.append('ai', file);
    if (readyisready) {
        showNotification('후처리가 완료되었습니다.', 'success');
    }

    try {
        const response = await fetch('/upload_ai', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            transcriptData = result;
            updateProgress(80);
            showNotification('ai 요약이 완료되었습니다.', 'success');
            isVideoUploaded = true; 
            checkAllUploadsComplete(); 
        } else {
            showError('ai 요약 실패: ' + result.error);
        }
    } catch (error) {
        showError('ai 요약 중 오류가 발생했습니다: ' + error.message);
    } finally {
        hideLoading();
    }
}

// PDF 서버 업로드
async function uploadPDFToServer(file) {
    showLoading('PDF를 업로드하고 처리하는 중...');
    
    const formData = new FormData();
    formData.append('pdf', file);
    
    try {
        const response = await fetch('/upload_pdf', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            pdfData = result;
            updateProgress(50);
            showNotification('PDF 업로드 및 처리가 완료되었습니다.', 'success');
            isPDFUploaded = true;  // ✅ 상태 업데이트
            checkAllUploadsComplete(); // ✅ 체크
        } else {
            showError('PDF 업로드 실패: ' + result.error);
        }
    } catch (error) {
        showError('PDF 업로드 중 오류가 발생했습니다: ' + error.message);
    } finally {
        hideLoading();
    }
}



// 진행도 업데이트
function updateProgress(percentage) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    progressFill.style.width = percentage + '%';
    
    const messages = {
        10: '비디오 파일이 업로드되었습니다. mp3로 변환이 완료되었습니다.',
        30: 'PDF 파일이 업로드중입니다.',
        50: '이미지 추출이 완료되었습니다.',
        80: '전처리를 시작합니다. 10분 기다려주세요',
        100: '모든 처리가 완료되었습니다!'
    };
    
    progressText.textContent = messages[percentage] || `처리 중... ${percentage}%`;
    
    // 처리 완료 시 애니메이션 제거
    if (percentage === 100) {
        progressFill.classList.remove('processing');
    }
}

// 진행바 처리 중 상태 시작
function startProgressProcessing(message) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    progressFill.classList.add('processing');
    
    // 동적 텍스트 효과
    let dots = '';
    const interval = setInterval(() => {
        dots = dots.length >= 3 ? '' : dots + '.';
        progressText.textContent = message + dots;
    }, 500);
    
    // interval ID를 저장해서 나중에 정리할 수 있도록
    progressFill.dataset.intervalId = interval;
}

// 진행바 처리 중 상태 종료
function stopProgressProcessing() {
    const progressFill = document.getElementById('progress-fill');
    
    progressFill.classList.remove('processing');
    
    // 동적 텍스트 효과 정리
    const intervalId = progressFill.dataset.intervalId;
    if (intervalId) {
        clearInterval(intervalId);
        delete progressFill.dataset.intervalId;
    }
}





// 로딩 표시
function showLoading(message = '처리 중입니다...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.querySelector('.loading-text');
    
    loadingText.textContent = message;
    loadingOverlay.style.display = 'flex';
}

// 로딩 숨김
function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// 에러 표시
function showError(message) {
    showNotification('오류: ' + message, 'error');
    console.error(message);
}

// 알림 표시 (토스트 스타일)
function showNotification(message, type = 'info') {
    // 기존 알림이 있으면 제거
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // 스타일 설정
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-size: 14px;
        font-weight: 500;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    
    // 타입에 따른 색상 설정
    switch(type) {
        case 'error':
            notification.style.backgroundColor = '#dc3545';
            break;
        case 'warning':
            notification.style.backgroundColor = '#ffc107';
            notification.style.color = '#212529';
            break;
        case 'success':
            notification.style.backgroundColor = '#28a745';
            break;
        default:
            notification.style.backgroundColor = '#007bff';
    }
    
    document.body.appendChild(notification);
    
    // 애니메이션으로 나타나기
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // 3초 후 사라지기
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

function openTab(event, tabId) {
    // 모든 탭 콘텐츠 숨기기
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.remove('active'));

    // 모든 탭 링크 비활성화
    const tabLinks = document.querySelectorAll('.tab-link');
    tabLinks.forEach(link => link.classList.remove('active'));

    // 클릭된 탭 활성화
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

// 시간 포맷팅
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// 이미지 모달 열기 (3D 효과용)
function openImageModal(imageSrc, title) {
    // 모달 창 생성
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 2000;
        cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = imageSrc;
    img.alt = title;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transform: scale(0.8);
        transition: transform 0.3s ease;
    `;
    
    // 애니메이션 효과
    setTimeout(() => {
        img.style.transform = 'scale(1)';
    }, 10);
    
    modal.appendChild(img);
    document.body.appendChild(modal);
    
    // 클릭 시 모달 닫기
    modal.addEventListener('click', () => {
        img.style.transform = 'scale(0.8)';
        setTimeout(() => {
            document.body.removeChild(modal);
        }, 300);
    });
}

// 드래그 앤 드롭 기능
function initializeDragAndDrop() {
    const uploadSections = document.querySelectorAll('.file-upload');
    
    uploadSections.forEach(section => {
        section.addEventListener('dragover', (e) => {
            e.preventDefault();
            section.style.backgroundColor = '#e3f2fd';
        });
        
        section.addEventListener('dragleave', (e) => {
            e.preventDefault();
            section.style.backgroundColor = '';
        });
        
        section.addEventListener('drop', (e) => {
            e.preventDefault();
            section.style.backgroundColor = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const isVideo = section.id === 'video-upload';
                
                if (isVideo && file.type.startsWith('video/')) {
                    document.getElementById('video-input').files = files;
                    handleVideoUpload({ target: { files: [file] } });
                } else if (!isVideo && file.type === 'application/pdf') {
                    document.getElementById('pdf-input').files = files;
                    handlePDFUpload({ target: { files: [file] } });
                }
            }
        });
    });
}

// 비디오 업로드 추가
function addVideoUpload() {
    const container = document.getElementById('video-uploads-container');
    const currentCount = container.children.length;
    
    if (currentCount >= MAX_VIDEOS) {
        showNotification(`비디오는 최대 ${MAX_VIDEOS}개까지만 업로드 가능합니다.`, 'warning');
        return;
    }
    
    const newUploadItem = document.createElement('div');
    newUploadItem.className = 'file-upload-item';
    newUploadItem.innerHTML = `
        <input type="file" class="video-input" accept=".mp4,.avi,.mov,.mkv" style="display: none;">
        <button class="file-type-btn" onclick="this.previousElementSibling.click()">
            Mp4 파일 업로드
        </button>
        <button class="remove-file-btn" onclick="removeFileUpload(this)">×</button>
    `;
    
    container.appendChild(newUploadItem);
    setupFileUploadEvents();
    updateAddButtonState();
}

// PDF 업로드 추가
function addPDFUpload() {
    const container = document.getElementById('pdf-uploads-container');
    const currentCount = container.children.length;
    
    if (currentCount >= MAX_PDFS) {
        showNotification(`PDF는 최대 ${MAX_PDFS}개까지만 업로드 가능합니다.`, 'warning');
        return;
    }
    
    const newUploadItem = document.createElement('div');
    newUploadItem.className = 'file-upload-item';
    newUploadItem.innerHTML = `
        <input type="file" class="pdf-input" accept=".pdf" style="display: none;">
        <button class="file-type-btn" onclick="this.previousElementSibling.click()">
            PDF 파일 업로드
        </button>
        <button class="remove-file-btn" onclick="removeFileUpload(this)">×</button>
    `;
    
    container.appendChild(newUploadItem);
    setupFileUploadEvents();
    updateAddButtonState();
}

// 파일 업로드 아이템 제거
function removeFileUpload(button) {
    const uploadItem = button.parentElement;
    const container = uploadItem.parentElement;
    
    // 첫 번째 아이템이 아닌 경우에만 제거 가능
    if (container.children.length > 1) {
        uploadItem.remove();
        updateAddButtonState();
        updateRemoveButtonsVisibility();
    }
}

// + 버튼 상태 업데이트
function updateAddButtonState() {
    const videoContainer = document.getElementById('video-uploads-container');
    const pdfContainer = document.getElementById('pdf-uploads-container');
    const addVideoBtn = document.getElementById('add-video-btn');
    const addPdfBtn = document.getElementById('add-pdf-btn');
    
    // 비디오 + 버튼 상태
    if (videoContainer.children.length >= MAX_VIDEOS) {
        addVideoBtn.disabled = true;
    } else {
        addVideoBtn.disabled = false;
    }
    
    // PDF + 버튼 상태
    if (pdfContainer.children.length >= MAX_PDFS) {
        addPdfBtn.disabled = true;
    } else {
        addPdfBtn.disabled = false;
    }
}

// 제거 버튼 가시성 업데이트
function updateRemoveButtonsVisibility() {
    const videoContainer = document.getElementById('video-uploads-container');
    const pdfContainer = document.getElementById('pdf-uploads-container');
    
    // 비디오 제거 버튼
    const videoRemoveBtns = videoContainer.querySelectorAll('.remove-file-btn');
    videoRemoveBtns.forEach((btn, index) => {
        btn.style.display = videoContainer.children.length > 1 ? 'flex' : 'none';
    });
    
    // PDF 제거 버튼
    const pdfRemoveBtns = pdfContainer.querySelectorAll('.remove-file-btn');
    pdfRemoveBtns.forEach((btn, index) => {
        btn.style.display = pdfContainer.children.length > 1 ? 'flex' : 'none';
    });
}

function updateCurrentPage(audio, slider, timeDisplay) {
    const currentPageDisplay = document.getElementById('current-page-display'); // 현재 페이지 표시 요소

    audio.addEventListener('timeupdate', () => {
        const currentTime = audio.currentTime;
        slider.value = (currentTime / audio.duration) * 100 || 0;
        timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(audio.duration)}`;

        // 현재 재생 시간 서버로 전송
        fetch('/current_page', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ current_time: currentTime })
        })
        .then(response => response.json())
        .then(data => {

            const currentPage = data.current_page;
            ///////// 이사이가 왜 안되는지 모르겠어서 올려봄
            const slideImageContainer = document.getElementById('slide-image-container');
            const imagePath = `/processed/${currentPage}.png`; // PDF 이름 없이 경로 생성

            console.log(`Generated image path: ${imagePath}`); // 디버깅 로그

            // 기존 이미지 제거
            slideImageContainer.innerHTML = '';

            // 새 이미지 추가
            const img = document.createElement('img');
            img.src = imagePath;
            img.alt = `Page ${currentPage}`;
            img.style.maxWidth = '100%';
            img.style.height = 'auto';

            img.onerror = () => console.error(`Failed to load image: ${imagePath}`); // 이미지 로드 실패 디버깅 로그

            slideImageContainer.appendChild(img);
            ////// 이사이가 왜 안되는지 모르겠어서 올려봄





            if (currentPage !== null) {
                currentPageDisplay.textContent = '현재 페이지: 111';//`현재 페이지: ${currentPage}`;
            } else {
                currentPageDisplay.textContent = '현재 페이지: -';
            }
        })
        .catch(error => console.error('Error fetching current page:', error));
    });
}
/*
function updateSlideImage(currentPage) {
    console.log(`Updating slide image: currentPage=${currentPage}`); // 디버깅 로그

    const slideImageContainer = document.getElementById('slide-image-container');
    const imagePath = `/get_processed/${currentPage}.png`; // PDF 이름 없이 경로 생성

    console.log(`Generated image path: ${imagePath}`); // 디버깅 로그

    // 기존 이미지 제거
    slideImageContainer.innerHTML = '';

    // 새 이미지 추가
    const img = document.createElement('img');
    img.src = imagePath;
    img.alt = `Page ${currentPage}`;
    img.style.maxWidth = '100%';
    img.style.height = 'auto';

    img.onerror = () => console.error(`Failed to load image: ${imagePath}`); // 이미지 로드 실패 디버깅 로그

    slideImageContainer.appendChild(img);
}
*/

// 서버에서 전달된 transcript_data.segments를 JavaScript 변수로 저장
const transcriptSegments = JSON.parse(document.getElementById('transcript-data').textContent);

// 대사 스크립트를 렌더링하는 함수
function renderTranscript(page) {
    console.log('Transcript Segments:', transcriptSegments);
    console.log("page 값:", page, "타입:", typeof page);
    //console.log("총 세그먼트 수:", transcriptData.segments.length);
    const transcriptContainer = document.getElementById('transcript-content');
    // 현재 페이지에 해당하는 세그먼트 필터링
    const filteredSegments = transcriptSegments.filter(segment => segment.page === Number(page));

    // 기존 내용 제거
    transcriptContainer.innerHTML = '';

    // 필터링된 세그먼트 렌더링
    if (filteredSegments.length > 0) {
        const segmentsDiv = document.createElement('div');
        segmentsDiv.className = 'segments';

        filteredSegments.forEach(segment => {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'segment';
            segmentDiv.setAttribute('data-start', segment.start);
            segmentDiv.setAttribute('data-end', segment.end);

            const timestamp = document.createElement('span');
            timestamp.className = 'timestamp';
            timestamp.textContent = `[${segment.start.toFixed(1)}s - ${segment.end.toFixed(1)}s]`;

            const text = document.createElement('span');
            text.className = 'segment-text';
            text.textContent = segment.text;

            segmentDiv.appendChild(timestamp);
            segmentDiv.appendChild(text);
            segmentsDiv.appendChild(segmentDiv);
        });

        transcriptContainer.appendChild(segmentsDiv);
    } else {
        transcriptContainer.innerHTML = '<p>해당 페이지에 대사가 없습니다.</p>';
    }
}

// 페이지 변경 시 호출되는 함수
window.updatePage = function(newPage) {
    currentPage = newPage;
    renderTranscript(currentPage);
}

function loadOpinion(page) {
    return fetch(`/get_opinion?page=${page}`)
        .then(response => response.json())
        .then(data => {
            console.log("받은 data:", data);  // 여기에 response 키가 있어야 함
            return data.response;  // ✅ 이 키 이름이 Flask의 jsonify 키랑 일치해야 함
        })
        .catch(error => {
            console.error("loadOpinion 에러:", error);
            return "오류 발생";
        });
}


/*
async function renderai(page) {
    const gemma_response = await loadOpinion(page);
    text.textContent = gemma_response;
}
*/
async function renderai(page) {
    console.log("page 값:", page, "타입:", typeof page);
    const aiContainer = document.getElementById('ai-content');
    aiContainer.innerHTML = '';

    // 비동기 데이터 받아오기
    const gemma_response = await loadOpinion(page);
    console.log("gemma_response :",gemma_response,"타입",typeof gemma_response)
    // 필터링된 세그먼트 렌더링
    const segmentsDiv = document.createElement('div');
    segmentsDiv.className = 'segments';

    const segmentDiv = document.createElement('div');

    // const page_number = document.createElement('span');
    // page_number.className = 'page_number';
    // page_number.textContent = `${page}`;

    const text = document.createElement('span');
    text.className = 'segment-text';
    text.textContent = gemma_response;  // 응답 사용

    // segmentDiv.appendChild(page_number);
    segmentDiv.appendChild(text);
    segmentsDiv.appendChild(segmentDiv);

    aiContainer.appendChild(segmentsDiv);
}


// 페이지 변경 시 호출되는 함수
window.updatePage_ai = function(newPage) {
    currentPage = newPage;
    renderai(currentPage);
}



document.addEventListener('DOMContentLoaded', function () {
    // 모든 segment 요소에 클릭 이벤트 리스너 추가
    const segments = document.querySelectorAll('.segment');
    const audio = document.getElementById('audio-player');
    const slider = document.getElementById('timeline-slider');
    const tooltip = document.getElementById('slider-tooltip');
    const timeDisplay = document.getElementById('time-display');

    console.log('slider:', slider, 'tooltip:', tooltip);
    if (!tooltip) {
    console.error('tooltip 요소를 찾을 수 없습니다.');
    return;
  }
    function formatTime(sec) {
        if (isNaN(sec)) return '0:00';
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
    if (!audio || !slider) {
        console.error('필수 DOM 요소를 찾을 수 없습니다.');
        return;
    }
    segments.forEach(segment => {
        segment.addEventListener('click', function () {
            // data-start 속성에서 시작 시간 가져오기
            const startTime = parseFloat(this.getAttribute('data-start'));

            if (!isNaN(startTime) && audio) {
                // 오디오 현재 시간을 클릭한 segment의 시작 시간으로 설정
                audio.currentTime = startTime;

                // 슬라이더 위치도 업데이트
                if (audio.duration) {
                    slider.value = (startTime / audio.duration) * 100;
                }

                // 시각적 피드백을 위해 활성 segment 표시
                segments.forEach(s => s.classList.remove('active-segment'));
                this.classList.add('active-segment');
            }
        });
    });
    slider.addEventListener('mousemove', e => {
        if (!audio.duration) return;

        /* 슬라이더 위에서의 x 좌표(0~1) 계산 */
        const rect = slider.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        const seconds = percent * audio.duration;
        console.log('tooltip seconds:', seconds);
        /* 툴팁 위치·내용 갱신 */
        tooltip.textContent = formatTime(seconds);
        tooltip.style.left = `${e.clientX-25}px`;});
    slider.addEventListener('mouseenter', () => {
        if (!tooltip) return;
        tooltip.hidden = false;
        tooltip.classList.add('show');
    });
    slider.addEventListener('mouseleave', () => {
        tooltip.classList.remove('show');
        /* transition 종료 후 숨김 */
        setTimeout(() => tooltip.hidden = true, 150);
    });
});

// let _imageFiles = [];            // 이미지 파일명 배열
// let _imageBasePath = '/output_images/'; // 이미지 파일 경로 기본값

/* 파일명 파싱: 뒤에서 3개 토큰 사용 */
function parseFilename(fname) {
  const base = fname.replace(/\.[^.]+$/, "");
  const parts = base.split("_");
  if (parts.length < 3) {
    return { title: base, page: null, index: null, tag: null };
  }
  const tag = parts[parts.length - 1];
  const idx = parts[parts.length - 2];
  const page = parts[parts.length - 3];
  const title = parts.slice(0, parts.length - 3).join("_") || "";
  return { title, page, index: idx, tag };
}

/* 태그 3 클릭 처리: 필요하면 커스터마이즈 */
/*
function onTag3Click(fileInfo, filename, event) {
  console.log("Tag3 클릭:", filename, fileInfo);

  // TODO: 여기서 3D 뷰어 열기 / API 호출 등 추가 로직 넣기
  // 예시: window.open(`/3d-view?file=${encodeURIComponent(filename)}`)
  const customEvent = new CustomEvent('gallery:tag3click', {
    detail: { filename, fileInfo }
  });
  window.dispatchEvent(customEvent);
}
*/
//살짝 변경해보자

async function onTag3Click(fileInfo, filename, event) {
  console.log("Tag3 클릭:", filename, fileInfo);

  // 3D 모델 생성 상태 표시
  const statusMessage = document.createElement('div');
  statusMessage.textContent = '3D 모델 생성 중...';
  statusMessage.style.color = 'blue';
  document.body.appendChild(statusMessage);

  try {
    // Flask API 호출
    const response = await fetch('/generate_3d', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ filename })
    });

    const result = await response.json();

    if (result.success) {
      statusMessage.textContent = '3D 모델 생성 완료! 새로운 창을 열고 있습니다.';
      statusMessage.style.color = 'green';

      // 새로운 창 열기
      window.open('https://imagetostl.com/kr/view-fbx-online', '_blank');
    } else {
      statusMessage.textContent = `오류: ${result.error}`;
      statusMessage.style.color = 'red';
    }
  } catch (error) {
    statusMessage.textContent = `요청 중 오류가 발생했습니다: ${error.message}`;
    statusMessage.style.color = 'red';
  }
}
/* 갤러리 렌더링 함수 (내부 사용) */
function _renderGalleryForPage(pageToShow) {
  const gallery = document.querySelector(".image-gallery");
  if (!gallery) {
    console.warn("image-gallery element not found.");
    return;
  }
  gallery.innerHTML = "";

  const filesToShow = _imageFiles.filter(fname => {
    const info = parseFilename(fname);
    if (info.page == null) return false;
    return String(info.page) === String(pageToShow);
  });

  filesToShow.forEach(fname => {
    const info = parseFilename(fname);
    const item = document.createElement("div");
    item.className = "gallery-item";

    const img = document.createElement("img");
    img.src = `${_imageBasePath.replace(/\/$/, '')}/${fname}`;
    img.alt = fname;

    const meta = document.createElement("div");
    meta.className = "item-meta";

    if (String(info.tag) === "3") {
      const btn = document.createElement("button");
      btn.className = "gallery-button";
      item.appendChild(img);
      item.appendChild(meta);
      btn.innerText = "3d화하기";
      btn.addEventListener("click", (ev) => onTag3Click(info, img.src, ev));
      item.appendChild(btn);
    } else {
      item.appendChild(img);
      item.appendChild(meta);
      item.classList.add("gallery-clickable");
      item.addEventListener("click", () => {
        // 일반 이미지 클릭 이벤트: 필요시 커스터마이즈
        const customEvent = new CustomEvent('gallery:imageclick', {
          detail: { filename: fname, fileInfo: info }
        });
        window.dispatchEvent(customEvent);
      });
    }
    if (String(info.tag) === "3" || String(info.tag) === "2") {
        gallery.appendChild(item);
    }
  });

  if (!gallery.children.length) {
    const empty = document.createElement("div");
    empty.style.padding = "20px";
    empty.innerText = "해당 페이지에 표시할 이미지가 없습니다.";
    gallery.appendChild(empty);
  }
}

/* 외부에서 파일 목록을 설정 */
function setImageFiles(files) {
  if (!Array.isArray(files)) {
    console.error("setImageFiles expects an array of filenames.");
    return;
  }
  _imageFiles = files.slice(); // 복사
}

/* 외부에서 베이스 경로 설정(선택) */
function setImageBasePath(path) {
  if (typeof path === 'string' && path.length) {
    _imageBasePath = path;
  }
}

/* 초기화 유틸: 한 번에 설정 */
function initGallery({ files = [], basePath = '/output_images/' } = {}) {
  setImageFiles(files);
  setImageBasePath(basePath);
  // 초기 렌더를 원치 않으므로 여기서는 렌더링하지 않음.
}

/* 외부에서 페이지 번호로 갤러리 업데이트 호출 */
function updateGallery(currentPage) {
  if (currentPage == null) {
    console.warn("updateGallery: currentPage is null/undefined");
    return;
  }
  _renderGalleryForPage(currentPage);
}

/* 전역 노출 */
window.initGallery = initGallery;
window.setImageFiles = setImageFiles;
window.setImageBasePath = setImageBasePath;
window.updateGallery = updateGallery;
// --- 자동 초기화 (템플릿 주입 방식 지원) ---
document.addEventListener("DOMContentLoaded", () => {
  // 서버에서 주입한 변수 사용 가능하면 initGallery 호출
  if (window.imageFilesFromServer && Array.isArray(window.imageFilesFromServer)) {
    // 기본 basePath 를 서버에서 주입했다면 사용, 아니면 기본 유지
    const basePath = window.imageBasePathFromServer || _imageBasePath;
    // initGallery는 기존 정의 사용
    initGallery({ files: window.imageFilesFromServer, basePath });
    // (선택) 페이지가 이미 존재하면 자동 렌더링 할 수도 있음.
    // 예를 들어 currentPage 변수를 템플릿에서 내려주면 바로 렌더:
    // if (typeof window.currentPageFromServer !== 'undefined') {
    //   updateGallery(window.currentPageFromServer);
    // }
  }
  playToggle.click();
});
