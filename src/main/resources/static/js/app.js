const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusIndicator = document.getElementById('status-indicator');
const predictedSignElement = document.getElementById('predicted-sign');
const confidenceElement = document.getElementById('confidence-level');

let isStreaming = false;
let stream = null;
let sequenceNumber = 0;
let isProcessing = false;

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const signTextInput = document.getElementById('sign-text-input');
const playSignBtn = document.getElementById('play-sign-btn');
const signsList = document.getElementById('signs-list');

// Configuration
const API_URL = '/api/sign-language'; // Relative path with context path
const FPS = 10;
const INTERVAL_MS = 1000 / FPS;

// Check backend health on load
checkBackendHealth();

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Tab Switching Logic
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');

        // Update Buttons
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update Content
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === targetTab) {
                content.classList.add('active');
            }
        });

        // Specific actions on tab change
        if (targetTab === 'sign-to-text') {
            updateStatus('System Ready', 'active');
        } else {
            updateStatus('Avatar View Active', 'active');
            // Ensure avatar is loaded and sized correctly
            if (window.AvatarRenderer) {
                window.AvatarRenderer.init();
                // We need to wait for the DOM to update the display style before resizing
                setTimeout(() => {
                    window.AvatarRenderer.handleResize();
                }, 100);
            }
            fetchAvailableSigns();
        }
    });
});

// Text to Sign Handlers
playSignBtn.addEventListener('click', () => {
    const text = signTextInput.value.trim();
    if (text) {
        playSignAnimation(text);
    }
});

signTextInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        playSignAnimation(signTextInput.value.trim());
    }
});

async function checkBackendHealth() {
    try {
        updateStatus('Connecting...', 'active');
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();

        if (data.status === 'UP' && data.pythonServiceStatus === 'UP') {
            updateStatus('System Ready', 'active');
            console.log('Backend connected:', data);
        } else {
            console.warn('Backend status:', data);
            updateStatus('Python Service Disconnected', 'error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatus('Backend Unavailable', 'error');
    }
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            }
        });

        videoElement.srcObject = stream;

        videoElement.onloadedmetadata = () => {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            isStreaming = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            updateStatus('Camera Active', 'active');
            requestAnimationFrame(processFrame);
        };
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please allow camera permissions.');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
        isStreaming = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        updateStatus('Camera Stopped');

        // Reset sequence on backend
        fetch(`${API_URL}/reset`, { method: 'POST' }).catch(console.error);
    }
}

async function processFrame(timestamp) {
    if (!isStreaming) return;

    // Draw frame to canvas
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    // Rate limiting: only process if not currently waiting for a response
    // (Or use simple throttle based on timestamp if prefer fixed FPS)
    if (!isProcessing) {
        sendFrameToBackend();
    }

    requestAnimationFrame(processFrame);
}

async function sendFrameToBackend() {
    isProcessing = true;

    try {
        const frameData = canvasElement.toDataURL('image/jpeg', 0.8);

        // Remove 'data:image/jpeg;base64,' prefix is handled by common libraries, 
        // but our backend might expect pure base64 or full data URI.
        // Looking at Python code: `if 'base64,' in frame_data: frame_data = frame_data.split('base64,')[1]`
        // So sending full data URI is fine.

        const payload = {
            frameData: frameData,
            sequenceNumber: sequenceNumber++
        };

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const result = await response.json();
            updateUI(result);
        } else {
            console.warn('Prediction error:', response.status);
        }

    } catch (error) {
        console.error('Error sending frame:', error);
    } finally {
        isProcessing = false;
        // Add small delay to prevent saturating the network/CPU if response is too fast
        // setTimeout(() => { isProcessing = false; }, 50); 
    }
}

function updateUI(result) {
    if (result.success) {
        // Update Prediction
        if (result.predictedSign) {
            predictedSignElement.textContent = result.mappedWord || result.predictedSign;
        }

        // Update Confidence
        if (result.confidence) {
            const percentage = Math.round(result.confidence * 100);
            confidenceElement.textContent = `${percentage}%`;

            // Visual feedback on confidence
            if (percentage > 80) {
                confidenceElement.style.color = 'var(--success-color)';
            } else {
                confidenceElement.style.color = 'var(--primary-color)';
            }
        }

        // Handling 'requiresMoreFrames' or stabilization messages
        if (result.message && result.message.includes('stabilizing')) {
            statusIndicator.textContent = 'Stabilizing...';
        } else if (result.requiresMoreFrames) {
            statusIndicator.textContent = 'Building Sequence...';
        } else {
            statusIndicator.textContent = 'Active';
        }
    }
}

function updateStatus(message, type) {
    statusIndicator.textContent = message;
    statusIndicator.className = 'status-indicator'; // Reset
    if (type) statusIndicator.classList.add(type);
}

// Text to Sign Helpers
async function fetchAvailableSigns() {
    try {
        const response = await fetch(`${API_URL}/dictionary`);
        if (response.ok) {
            const signs = await response.json();
            signsList.innerHTML = '';
            signs.forEach(sign => {
                const tag = document.createElement('span');
                tag.className = 'sign-tag';
                tag.textContent = sign;
                tag.onclick = () => {
                    signTextInput.value = sign;
                    playSignAnimation(sign);
                };
                signsList.appendChild(tag);
            });
        }
    } catch (error) {
        console.error('Error fetching signs:', error);
    }
}

function playSignAnimation(text) {
    console.log('Playing animation for:', text);
    // This will communicate with avatar-renderer.js
    if (window.AvatarRenderer && window.AvatarRenderer.playAnimation) {
        window.AvatarRenderer.playAnimation(text);
    } else {
        alert('Avatar renderer not initialized or animation not found.');
    }
}
