const videoElement         = document.getElementById('webcam');
const canvasElement        = document.getElementById('output_canvas');
const canvasCtx            = canvasElement.getContext('2d');
const startBtn             = document.getElementById('start-btn');
const stopBtn              = document.getElementById('stop-btn');
const statusIndicator      = document.getElementById('status-indicator');
const predictedSignElement = document.getElementById('predicted-sign');
const confidenceElement    = document.getElementById('confidence-level');

let isStreaming    = false;
let stream         = null;
let sequenceNumber = 0;
let pendingRequest = false;
let lastSentTime   = 0;

// Tab elements
const tabBtns       = document.querySelectorAll('.tab-btn');
const tabContents   = document.querySelectorAll('.tab-content');
const signTextInput = document.getElementById('sign-text-input');
const playSignBtn   = document.getElementById('play-sign-btn');
const signsList     = document.getElementById('signs-list');

// ── Configuration ─────────────────────────────────────────────────────────
const JAVA_API_URL = '/api/sign-language'; 
const FLASK_URL    = '/api/python'; // ROUTED VIA VERCEL
const FPS          = 15;
const INTERVAL_MS  = 1000 / FPS;
const JPEG_QUALITY = 0.5; // Small payload
const CAPTURE_W    = 320;
const CAPTURE_H    = 240;
// ─────────────────────────────────────────────────────────────────────────

// Offscreen canvas for downscaled capture
const offscreen    = document.createElement('canvas');
offscreen.width    = CAPTURE_W;
offscreen.height   = CAPTURE_H;
const offscreenCtx = offscreen.getContext('2d');

// Progress bar (injected once)
const progressBar = (() => {
    let el = document.getElementById('sequence-progress-fill');
    if (!el) {
        const wrapper = document.createElement('div');
        wrapper.id = 'progress-wrapper';
        wrapper.style.cssText = `margin:6px 0;display:flex;align-items:center;gap:8px;font-size:0.8rem;color:var(--text-secondary,#888)`;
        const track = document.createElement('div');
        track.style.cssText = `flex:1;height:5px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden`;
        const fill = document.createElement('div');
        fill.id = 'sequence-progress-fill';
        fill.style.cssText = `height:100%;width:0%;background:linear-gradient(90deg,#6c63ff,#48cfad);border-radius:4px;transition:width 0.1s ease`;
        track.appendChild(fill);
        const lbl = document.createElement('span');
        lbl.id = 'sequence-progress-label';
        lbl.textContent = '';
        wrapper.append(track, lbl);
        const parent = predictedSignElement?.parentElement;
        if (parent) parent.insertBefore(wrapper, predictedSignElement);
        el = fill;
    }
    return el;
})();

function setProgress(cur, total) {
    const pct = total > 0 ? Math.round((cur / total) * 100) : 0;
    if (progressBar) progressBar.style.width = `${pct}%`;
    const lbl = document.getElementById('sequence-progress-label');
    if (lbl) lbl.textContent = total > 0 && cur < total ? `${cur}/${total}` : '';
}

// Check backend health on load
checkBackendHealth();

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        tabContents.forEach(c => {
            c.classList.remove('active');
            if (c.id === targetTab) c.classList.add('active');
        });
        if (targetTab === 'sign-to-text') {
            updateStatus('System Ready', 'active');
        } else {
            updateStatus('Avatar View Active', 'active');
            if (window.AvatarRenderer) {
                window.AvatarRenderer.init();
                setTimeout(() => window.AvatarRenderer.handleResize(), 100);
            }
            fetchAvailableSigns();
        }
    });
});

playSignBtn.addEventListener('click', () => {
    const text = signTextInput.value.trim();
    if (text) playSignAnimation(text);
});
signTextInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') playSignAnimation(signTextInput.value.trim());
});

async function checkBackendHealth() {
    try {
        updateStatus('Connecting...', 'active');
        const response = await fetch(`${JAVA_API_URL}/health`);
        const data = await response.json();
        if (data.status === 'UP' && data.pythonServiceStatus === 'UP') {
            updateStatus('System Ready', 'active');
        } else {
            updateStatus('Python Service Disconnected', 'error');
        }
    } catch {
        updateStatus('Backend Unavailable', 'error');
    }
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
            canvasElement.width  = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            isStreaming  = true;
            startBtn.disabled = true;
            stopBtn.disabled  = false;
            updateStatus('Camera Active', 'active');
            requestAnimationFrame(processFrame);
        };
    } catch {
        alert('Could not access camera. Please allow camera permissions.');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        videoElement.srcObject = null;
        isStreaming = false;
        startBtn.disabled = false;
        stopBtn.disabled  = true;
        updateStatus('Camera Stopped');
        setProgress(0, 0);
        fetch(`${FLASK_URL}/reset`, { method: 'POST' }).catch(console.error);
    }
}

function processFrame(timestamp) {
    if (!isStreaming) return;
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    if (!pendingRequest && (timestamp - lastSentTime >= INTERVAL_MS)) {
        lastSentTime = timestamp;
        sendFrameToBackend();
    }
    requestAnimationFrame(processFrame);
}

async function sendFrameToBackend() {
    pendingRequest = true;
    try {
        offscreenCtx.drawImage(videoElement, 0, 0, CAPTURE_W, CAPTURE_H);
        const frameData = offscreen.toDataURL('image/jpeg', JPEG_QUALITY);
        
        // DIRECT CALL TO FLASK (Zero Delay)
        const response  = await fetch(`${FLASK_URL}/predict`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ frameData, sequenceNumber: sequenceNumber++ })
        });
        
        if (response.ok) {
            const result = await response.json();
            updateUI(result);
        }
    } catch (e) {
        console.error('Direct Flask call error:', e);
    } finally {
        pendingRequest = false;
    }
}

function updateUI(result) {
    if (!result.success) return;

    // Warmup
    if (result.requiresMoreFrames) {
        if (result.progress != null) setProgress(result.progress, result.total || 30);
        const handHint = result.handDetected === false ? ' (Show Hand)' : ' (Wait...)';
        updateStatus('Initializing... ' + (result.progress || 0) + '/' + (result.total || 10) + handHint);
        return;
    }

    // Prediction
    if (result.predictedSign || result.mappedWord) {
        const word = result.mappedWord || result.predictedSign;
        predictedSignElement.textContent = word;
        const conf = result.confidence ? Math.round(result.confidence * 100) + '%' : '';
        confidenceElement.textContent = conf;
        setProgress(10, 10);
        updateStatus('✓ ' + word + ' (' + conf + ')', 'active');
    } else {
        // Low confidence — keep processing
        const handStatus = result.handDetected ? 'Hand Detected ✓' : 'Show Hand';
        const conf = result.confidence ? Math.round(result.confidence * 100) + '%' : '';
        confidenceElement.textContent = conf || '0%';
        updateStatus(handStatus + ' | Analyzing...');
        setProgress(0, 0);
    }
}

function updateStatus(message, type) {
    statusIndicator.textContent = message;
    statusIndicator.className   = 'status-indicator';
    if (type) statusIndicator.classList.add(type);
}

async function fetchAvailableSigns() {
    try {
        const response = await fetch(`${JAVA_API_URL}/dictionary`);
        if (response.ok) {
            const signs = await response.json();
            signsList.innerHTML = '';
            signs.forEach(sign => {
                const tag = document.createElement('span');
                tag.className   = 'sign-tag';
                tag.textContent = sign;
                tag.onclick     = () => { signTextInput.value = sign; playSignAnimation(sign); };
                signsList.appendChild(tag);
            });
        }
    } catch (e) { console.error('fetchAvailableSigns:', e); }
}

function playSignAnimation(text) {
    if (window.AvatarRenderer?.playAnimation) {
        window.AvatarRenderer.playAnimation(text);
    } else {
        alert('Avatar renderer not initialized.');
    }
}
