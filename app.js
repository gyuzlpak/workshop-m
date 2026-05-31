const BUILT_IN_TRACKS = [
  { id: "track-01", title: "Track 01", src: "./audio/track-01.mp3" },
  { id: "track-02", title: "Track 02", src: "./audio/track-02.mp3" },
  { id: "track-03", title: "Track 03", src: "./audio/track-03.mp3" },
];

const VISION_IMPORT_URLS = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest",
];
const WASM_URLS = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
];
const MODEL_URLS = [
  "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task",
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
];

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const els = {
  visual: document.querySelector("#visual"),
  playButton: document.querySelector("#playButton"),
  cameraButton: document.querySelector("#cameraButton"),
  fileInput: document.querySelector("#fileInput"),
  status: document.querySelector("#status"),
  player: document.querySelector("#player"),
  webcam: document.querySelector("#webcam"),
  overlay: document.querySelector("#overlay"),
  cameraPanel: document.querySelector(".camera-panel"),
  trackCaption: document.querySelector("#trackCaption"),
  volumeReadout: document.querySelector("#volumeReadout"),
};

const visualCtx = els.visual.getContext("2d", { alpha: false });
const overlayCtx = els.overlay.getContext("2d");

const state = {
  tracks: [...BUILT_IN_TRACKS],
  activeTrackId: BUILT_IN_TRACKS[0].id,
  hasTrack: false,
  isPlaying: false,
  cameraActive: false,
  handCount: 0,
  hasHandInput: false,
  targetVolume: 0.52,
  volume: 0.52,
  targetSpace: 0,
  space: 0,
  openness: 0,
  blobPhase: 0,
  modelReady: false,
  detectFrames: 0,
  lastStatusAt: 0,
  pendingFingerCount: 0,
  pendingFingerSince: 0,
  trackGestureCooldownUntil: 0,
  trackFeedbackUntil: 0,
};

let audio = null;
let visionTasks = null;
let handLandmarker = null;
let webcamStream = null;
let lastVideoTime = -1;
let animationHandle = null;
let detectHandle = null;
let lastModelError = null;
const artworkImages = new Map();

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function lerp(from, to, amount) {
  return from + (to - from) * amount;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getPalmCenter(hand) {
  return {
    x: (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5,
    y: (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5,
  };
}

function getHandMetrics(hand) {
  const tipIds = [4, 8, 12, 16, 20];
  const palmWidth = Math.max(0.025, distance(hand[5], hand[17]));
  let tipSpread = 0;
  let pairs = 0;

  for (let i = 0; i < tipIds.length; i += 1) {
    for (let j = i + 1; j < tipIds.length; j += 1) {
      tipSpread += distance(hand[tipIds[i]], hand[tipIds[j]]) / palmWidth;
      pairs += 1;
    }
  }

  const avgTipSpread = tipSpread / pairs;
  const avgTipToPalm =
    tipIds.reduce((sum, id) => sum + distance(hand[id], hand[0]) / palmWidth, 0) /
    tipIds.length;
  const openBySpread = clamp((avgTipSpread - 0.85) / 2.15, 0, 1);
  const openByPalm = clamp((avgTipToPalm - 1.15) / 1.15, 0, 1);
  const openness = clamp(openBySpread * 0.68 + openByPalm * 0.32, 0, 1);
  const cluster = clamp((2.05 - avgTipToPalm) / 1.05, 0, 1);
  const spread = clamp((avgTipSpread - 1.65) / 1.35, 0, 1);

  return {
    center: getPalmCenter(hand),
    openness,
    spread,
    cluster,
    avgTipSpread,
  };
}

function countExtendedFingers(hand) {
  const palmWidth = Math.max(0.025, distance(hand[5], hand[17]));
  const wrist = hand[0];
  const fingers = [
    { tip: 8, pip: 6 },
    { tip: 12, pip: 10 },
    { tip: 16, pip: 14 },
    { tip: 20, pip: 18 },
  ];
  let count = 0;

  fingers.forEach(({ tip, pip }) => {
    const extension = (distance(hand[tip], wrist) - distance(hand[pip], wrist)) / palmWidth;
    if (extension > 0.18) count += 1;
  });

  const thumbSpread = distance(hand[4], hand[9]) / palmWidth;
  const thumbExtension = (distance(hand[4], wrist) - distance(hand[3], wrist)) / palmWidth;
  if (thumbSpread > 1.08 && thumbExtension > 0.05) count += 1;

  return count;
}

function resizeCanvas(canvas, ctx) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function setStatus(text) {
  els.status.textContent = text;
}

function setTimedStatus(text, interval = 700) {
  const now = performance.now();
  if (now < state.trackFeedbackUntil && !text.startsWith("Track")) return;
  if (now - state.lastStatusAt < interval && els.status.textContent === text) return;

  state.lastStatusAt = now;
  setStatus(text);
}

function readAscii(bytes, start, length) {
  let result = "";
  for (let i = start; i < start + length && i < bytes.length; i += 1) {
    result += String.fromCharCode(bytes[i]);
  }
  return result;
}

function readSynchsafe(bytes, start) {
  return (
    (bytes[start] << 21) |
    (bytes[start + 1] << 14) |
    (bytes[start + 2] << 7) |
    bytes[start + 3]
  );
}

function readUint32(bytes, start) {
  return (
    (bytes[start] << 24) |
    (bytes[start + 1] << 16) |
    (bytes[start + 2] << 8) |
    bytes[start + 3]
  );
}

function loadArtworkImage(track) {
  if (!track.artworkUrl || artworkImages.has(track.id)) return;

  const image = new Image();
  image.onload = () => {
    artworkImages.set(track.id, image);
  };
  image.src = track.artworkUrl;
}

function updateTrackCaption() {
  const track = state.tracks.find((item) => item.id === state.activeTrackId);
  els.trackCaption.textContent = track ? track.title : "No track selected";
}

async function extractMp3Artwork(file) {
  const header = new Uint8Array(await file.slice(0, 10).arrayBuffer());
  if (readAscii(header, 0, 3) !== "ID3") return null;

  const tagSize = readSynchsafe(header, 6);
  const tagBytes = new Uint8Array(await file.slice(0, 10 + tagSize).arrayBuffer());
  const version = header[3];
  let offset = 10;

  while (offset + 10 < tagBytes.length) {
    const frameId = readAscii(tagBytes, offset, 4);
    if (!frameId.trim()) break;

    const frameSize = version === 4 ? readSynchsafe(tagBytes, offset + 4) : readUint32(tagBytes, offset + 4);
    const frameStart = offset + 10;
    const frameEnd = frameStart + frameSize;

    if (frameEnd > tagBytes.length || frameSize <= 0) break;

    if (frameId === "APIC") {
      let cursor = frameStart + 1;
      const mimeStart = cursor;

      while (cursor < frameEnd && tagBytes[cursor] !== 0) cursor += 1;

      const mimeType = readAscii(tagBytes, mimeStart, cursor - mimeStart) || "image/jpeg";
      cursor += 2;

      while (cursor < frameEnd && tagBytes[cursor] !== 0) cursor += 1;
      cursor += 1;

      const imageBytes = tagBytes.slice(cursor, frameEnd);
      if (!imageBytes.length) return null;

      return URL.createObjectURL(new Blob([imageBytes], { type: mimeType }));
    }

    offset = frameEnd;
  }

  return null;
}

function getCameraErrorMessage(error) {
  if (!window.isSecureContext) {
    return "Open with http://localhost:8000";
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    return "Camera requires HTTPS or localhost";
  }

  if (error?.name === "NotAllowedError") {
    return "Allow camera permission";
  }

  if (error?.name === "NotFoundError") {
    return "No camera found";
  }

  if (error?.name === "NotReadableError") {
    return "Camera is already in use";
  }

  return "Camera unavailable";
}

function getModelErrorMessage(error) {
  const message = String(error?.message || error || "");

  if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
    return "Hand model network failed";
  }

  if (message.includes("import")) {
    return "MediaPipe import failed";
  }

  if (message.includes("WebAssembly") || message.includes("wasm")) {
    return "MediaPipe wasm failed";
  }

  return "Hand model failed";
}

async function loadVisionTasks() {
  if (visionTasks) return visionTasks;

  let error = null;
  for (const url of VISION_IMPORT_URLS) {
    try {
      visionTasks = await import(url);
      return visionTasks;
    } catch (nextError) {
      error = nextError;
      console.error(`MediaPipe import failed: ${url}`, nextError);
    }
  }

  throw error;
}

function selectTrack(id) {
  const track = state.tracks.find((item) => item.id === id);
  if (!track) return;

  state.activeTrackId = id;
  state.hasTrack = true;
  els.player.src = track.src;
  els.player.load();
  loadArtworkImage(track);
  updateTrackCaption();
  setStatus(track.title);

  if (state.isPlaying) {
    playAudio();
  }
}

function selectTrackByFingerCount(fingerCount) {
  if (fingerCount < 1 || fingerCount > state.tracks.length) return;

  const nextTrack = state.tracks[fingerCount - 1];
  if (nextTrack.id === state.activeTrackId) return;

  selectTrack(nextTrack.id);
  state.trackFeedbackUntil = performance.now() + 950;
  setTimedStatus(`Track ${fingerCount}: ${nextTrack.title}`, 0);
}

function createImpulseResponse(audioContext, seconds = 2.2, decay = 3.4) {
  const rate = audioContext.sampleRate;
  const length = rate * seconds;
  const impulse = audioContext.createBuffer(2, length, rate);

  for (let channel = 0; channel < 2; channel += 1) {
    const data = impulse.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      const n = 1 - i / length;
      data[i] = (Math.random() * 2 - 1) * n ** decay;
    }
  }

  return impulse;
}

function setupAudio() {
  if (audio) return audio;

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContext();
  const source = context.createMediaElementSource(els.player);
  const volumeGain = context.createGain();
  const dryGain = context.createGain();
  const clusterFilter = context.createBiquadFilter();
  const clusterGain = context.createGain();
  const convolver = context.createConvolver();
  const reverbGain = context.createGain();
  const compressor = context.createDynamicsCompressor();

  clusterFilter.type = "lowpass";
  clusterFilter.frequency.value = 1200;
  clusterFilter.Q.value = 0.8;
  convolver.buffer = createImpulseResponse(context);
  reverbGain.gain.value = 0;
  clusterGain.gain.value = 0;
  dryGain.gain.value = 1;

  compressor.threshold.value = -18;
  compressor.knee.value = 24;
  compressor.ratio.value = 3;
  compressor.attack.value = 0.012;
  compressor.release.value = 0.18;

  source.connect(volumeGain);
  volumeGain.connect(dryGain);
  volumeGain.connect(clusterFilter);
  volumeGain.connect(convolver);
  clusterFilter.connect(clusterGain);
  dryGain.connect(compressor);
  clusterGain.connect(compressor);
  convolver.connect(reverbGain);
  reverbGain.connect(compressor);
  compressor.connect(context.destination);

  audio = {
    context,
    volumeGain,
    dryGain,
    clusterFilter,
    clusterGain,
    reverbGain,
  };

  return audio;
}

async function playAudio() {
  const engine = setupAudio();

  if (engine.context.state !== "running") {
    await engine.context.resume();
  }

  if (!state.hasTrack) {
    selectTrack(state.activeTrackId);
  }

  try {
    await els.player.play();
    state.isPlaying = true;
    els.playButton.textContent = "Pause";
  } catch {
    state.isPlaying = false;
    els.playButton.textContent = "Play";
    setStatus("Add an mp3 file");
  }
}

function pauseAudio() {
  els.player.pause();
  state.isPlaying = false;
  els.playButton.textContent = "Play";
}

async function initHandLandmarker() {
  if (handLandmarker) return handLandmarker;

  setStatus("Loading hand model");
  await loadVisionTasks();
  const { FilesetResolver, HandLandmarker } = visionTasks;

  for (const wasmUrl of WASM_URLS) {
    let fileset = null;

    try {
      fileset = await FilesetResolver.forVisionTasks(wasmUrl);
    } catch (error) {
      lastModelError = error;
      console.error(`MediaPipe wasm failed: ${wasmUrl}`, error);
      continue;
    }

    for (const modelUrl of MODEL_URLS) {
      for (const delegate of ["GPU", undefined]) {
        try {
          handLandmarker = await HandLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: modelUrl,
              ...(delegate ? { delegate } : {}),
            },
            runningMode: "VIDEO",
            numHands: 4,
            minHandDetectionConfidence: 0.32,
            minHandPresenceConfidence: 0.32,
            minTrackingConfidence: 0.32,
          });

          return handLandmarker;
        } catch (error) {
          lastModelError = error;
          console.error(`Hand model failed: ${modelUrl}`, error);
        }
      }
    }
  }

  throw lastModelError || new Error("Hand model failed");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("getUserMedia unavailable");
  }

  els.cameraButton.disabled = true;
  els.cameraButton.textContent = "Loading";
  els.cameraPanel.classList.add("is-visible");
  setStatus("Opening camera");

  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  els.webcam.srcObject = webcamStream;
  await els.webcam.play();
  await waitForVideoFrame();

  state.cameraActive = true;
  state.modelReady = false;
  state.detectFrames = 0;
  lastVideoTime = -1;
  els.cameraButton.textContent = "Stop";
  setStatus("Camera on");

  try {
    await initHandLandmarker();
    state.modelReady = true;
    setStatus("Model ready, show your palm");
    detectHands();
  } catch (error) {
    console.error(error);
    setStatus(`Camera on, ${getModelErrorMessage(error)}`);
  } finally {
    els.cameraButton.disabled = false;
  }
}

function stopCamera() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
  }

  webcamStream = null;
  state.cameraActive = false;
  state.handCount = 0;
  state.hasHandInput = false;
  state.pendingFingerCount = 0;
  state.pendingFingerSince = 0;
  state.trackGestureCooldownUntil = 0;
  state.trackFeedbackUntil = 0;
  state.modelReady = false;
  els.cameraButton.disabled = false;
  els.cameraButton.textContent = "Camera";
  els.cameraPanel.classList.remove("is-visible");
  cancelAnimationFrame(detectHandle);
  overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
}

function waitForVideoFrame() {
  if (els.webcam.videoWidth > 0 && els.webcam.videoHeight > 0) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    els.webcam.addEventListener("loadeddata", resolve, { once: true });
  });
}

function analyzeHands(hands) {
  if (!hands.length) {
    state.handCount = 0;
    state.hasHandInput = false;
    state.targetVolume = lerp(state.targetVolume, state.volume, 0.06);
    state.targetSpace = lerp(state.targetSpace, 0, 0.04);
    state.pendingFingerCount = 0;
    state.pendingFingerSince = 0;
    return;
  }

  let spaceTotal = 0;
  let opennessTotal = 0;
  let volumeTotal = 0;
  let trackFingerCount = 0;

  hands.forEach((hand) => {
    const metrics = getHandMetrics(hand);
    const fingerCount = countExtendedFingers(hand);

    volumeTotal += metrics.openness;
    spaceTotal += metrics.spread - metrics.cluster;
    opennessTotal += metrics.avgTipSpread;

    trackFingerCount = Math.max(trackFingerCount, fingerCount);
  });

  state.handCount = hands.length;
  state.hasHandInput = true;
  state.targetVolume = clamp(volumeTotal / hands.length, 0.03, 1);
  state.targetSpace = clamp(spaceTotal / hands.length, -1, 1);
  state.openness = opennessTotal / hands.length;
  updateTrackByFingerCount(trackFingerCount);
}

function updateTrackByFingerCount(fingerCount) {
  const now = performance.now();

  if (fingerCount < 1 || fingerCount > state.tracks.length) {
    state.pendingFingerCount = 0;
    state.pendingFingerSince = 0;
    return;
  }

  if (fingerCount !== state.pendingFingerCount) {
    state.pendingFingerCount = fingerCount;
    state.pendingFingerSince = now;
    return;
  }

  if (now < state.trackGestureCooldownUntil || now - state.pendingFingerSince < 260) return;

  selectTrackByFingerCount(fingerCount);
  state.trackGestureCooldownUntil = now + 900;
}

function drawHandOverlay(hands) {
  resizeCanvas(els.overlay, overlayCtx);
  const width = els.overlay.clientWidth;
  const height = els.overlay.clientHeight;

  overlayCtx.clearRect(0, 0, width, height);

  hands.forEach((hand) => {
    overlayCtx.lineWidth = 1.2;
    overlayCtx.strokeStyle = "rgba(248, 248, 245, 0.86)";
    overlayCtx.fillStyle = "rgba(248, 248, 245, 0.92)";

    HAND_CONNECTIONS.forEach(([a, b]) => {
      overlayCtx.beginPath();
      overlayCtx.moveTo(hand[a].x * width, hand[a].y * height);
      overlayCtx.lineTo(hand[b].x * width, hand[b].y * height);
      overlayCtx.stroke();
    });

    hand.forEach((point, index) => {
      overlayCtx.beginPath();
      overlayCtx.arc(point.x * width, point.y * height, index % 4 === 0 ? 3 : 2, 0, Math.PI * 2);
      overlayCtx.fill();
    });
  });

}

function drawTrackArtworks(centerX, centerY, baseRadius) {
  const activity = state.hasHandInput ? 1 : 0;
  const orbitX = baseRadius * (1.72 + state.volume * 0.28);
  const orbitY = baseRadius * (1.18 + state.volume * 0.16);
  const coverRadius = Math.max(18, Math.min(baseRadius * 0.16, 54));

  state.tracks.forEach((track, index) => {
    loadArtworkImage(track);

    const phase =
      state.blobPhase * ((0.04 + activity * 0.14) + index * 0.008) +
      index * ((Math.PI * 2) / Math.max(1, state.tracks.length));
    const x = centerX + Math.cos(phase) * orbitX;
    const y = centerY + Math.sin(phase) * orbitY;
    const isActive = track.id === state.activeTrackId;
    const radius = coverRadius * (isActive ? 1.18 : 0.9);
    const image = artworkImages.get(track.id);

    visualCtx.save();
    visualCtx.beginPath();
    visualCtx.arc(x, y, radius, 0, Math.PI * 2);
    visualCtx.clip();

    if (image) {
      const size = radius * 2;
      visualCtx.drawImage(image, x - radius, y - radius, size, size);
    } else {
      visualCtx.fillStyle = isActive ? "#050505" : "#d8d5cd";
      visualCtx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    }

    visualCtx.restore();
    visualCtx.lineWidth = isActive ? 3 : 1;
    visualCtx.strokeStyle = isActive ? "#050505" : "rgba(5, 5, 5, 0.28)";
    visualCtx.beginPath();
    visualCtx.arc(x, y, radius, 0, Math.PI * 2);
    visualCtx.stroke();
  });
}

function detectHands() {
  if (!state.cameraActive || !handLandmarker) return;

  if (els.webcam.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    try {
      if (els.webcam.currentTime !== lastVideoTime) {
        lastVideoTime = els.webcam.currentTime;
        state.detectFrames += 1;

        const result = handLandmarker.detectForVideo(els.webcam, performance.now());
        const hands = result.landmarks || [];

        analyzeHands(hands);
        drawHandOverlay(hands);

        if (hands.length) {
          setTimedStatus("Finger count selects track", 420);
        } else if (state.detectFrames > 10) {
          setTimedStatus("Model ready, show your palm", 900);
        }
      }
    } catch (error) {
      console.error(error);
      setTimedStatus("Hand detection retrying", 900);
    }
  }

  detectHandle = requestAnimationFrame(detectHands);
}

function applyAudioState() {
  if (!audio) return;

  const now = audio.context.currentTime;
  const spread = Math.max(0, state.space);
  const cluster = Math.max(0, -state.space);

  audio.volumeGain.gain.setTargetAtTime(state.volume, now, 0.045);
  audio.dryGain.gain.setTargetAtTime(1 - spread * 0.2 - cluster * 0.28, now, 0.08);
  audio.reverbGain.gain.setTargetAtTime(spread * 0.62, now, 0.1);
  audio.clusterGain.gain.setTargetAtTime(cluster * 0.52, now, 0.08);
  audio.clusterFilter.frequency.setTargetAtTime(1300 - cluster * 820, now, 0.08);
}

function drawVisual() {
  resizeCanvas(els.visual, visualCtx);

  const width = els.visual.clientWidth;
  const height = els.visual.clientHeight;
  const centerX = width * 0.5;
  const centerY = height * 0.51;
  const activity = state.hasHandInput ? 1 : 0;
  const volumeScale = 0.58 + state.volume * 0.62;
  const baseRadius = Math.min(width, height) * 0.245 * volumeScale;
  const spread = Math.max(0, state.space);
  const cluster = Math.max(0, -state.space);
  const pulse = 1 + (state.volume - 0.5) * (0.04 + activity * 0.06);
  const lobes = 3 + Math.round(spread * 3);
  const wobble = 0.025 + activity * 0.055 + spread * 0.08 + cluster * 0.03;

  state.blobPhase += activity ? 0.012 + state.volume * 0.006 : 0.0025;
  state.volume = lerp(state.volume, state.targetVolume, 0.075);
  state.space = lerp(state.space, state.targetSpace, 0.065);

  visualCtx.fillStyle = "#f8f8f5";
  visualCtx.fillRect(0, 0, width, height);
  visualCtx.fillStyle = "#050505";

  visualCtx.beginPath();
  for (let i = 0; i <= 220; i += 1) {
    const t = (i / 220) * Math.PI * 2;
    const organic =
      Math.sin(t * lobes + state.blobPhase) * wobble +
      Math.cos(t * 2 - state.blobPhase * 0.7) * (0.055 + cluster * 0.07);
    const horizontal = 1.26 + spread * 0.24 - cluster * 0.16;
    const vertical = 0.82 + cluster * 0.08;
    const radius = baseRadius * pulse * (1 + organic);
    const x = centerX + Math.cos(t) * radius * horizontal;
    const y = centerY + Math.sin(t) * radius * vertical;

    if (i === 0) visualCtx.moveTo(x, y);
    else visualCtx.lineTo(x, y);
  }
  visualCtx.closePath();
  visualCtx.fill();

  const dotBase = Math.min(width, height) * 0.038;
  const orbitX = baseRadius * (1.38 + spread * 0.45);
  const orbitY = baseRadius * (0.96 + cluster * 0.18);
  const dots = [
    { a: -1.72, r: 1.04 },
    { a: 0.05, r: 1.5 },
    { a: 1.5, r: 0.68 },
  ];

  dots.forEach((dot, index) => {
    const phase = dot.a + state.blobPhase * (0.22 + index * 0.035);
    const x = centerX + Math.cos(phase) * orbitX;
    const y = centerY + Math.sin(phase) * orbitY;
    const r = dotBase * dot.r * (0.86 + state.volume * 0.26);

    visualCtx.beginPath();
    visualCtx.arc(x, y, r, 0, Math.PI * 2);
    visualCtx.fill();
  });

  drawTrackArtworks(centerX, centerY, baseRadius);

  els.volumeReadout.textContent = Math.round(state.volume * 100);
  applyAudioState();

  animationHandle = requestAnimationFrame(drawVisual);
}

async function loadLocalFiles(files) {
  const newTracks = await Promise.all(
    Array.from(files)
      .filter((file) => file.type.startsWith("audio/"))
      .map(async (file, index) => ({
      id: `local-${Date.now()}-${index}`,
      artworkUrl: await extractMp3Artwork(file),
      title: file.name.replace(/\.[^/.]+$/, ""),
      src: URL.createObjectURL(file),
    }))
  );

  if (!newTracks.length) return;

  newTracks.forEach(loadArtworkImage);

  state.tracks = [...newTracks, ...state.tracks];
  state.activeTrackId = newTracks[0].id;
  selectTrack(newTracks[0].id);
}

function bindEvents() {
  els.playButton.addEventListener("click", () => {
    if (state.isPlaying) pauseAudio();
    else playAudio();
  });

  els.cameraButton.addEventListener("click", async () => {
    if (state.cameraActive) {
      stopCamera();
      return;
    }

    try {
      await startCamera();
    } catch (error) {
      state.cameraActive = false;
      els.cameraButton.disabled = false;
      els.cameraButton.textContent = "Camera";
      els.cameraPanel.classList.remove("is-visible");
      setStatus(getCameraErrorMessage(error));
    }
  });

  els.fileInput.addEventListener("change", async (event) => {
    await loadLocalFiles(event.target.files);
    event.target.value = "";
  });

  els.player.addEventListener("error", () => {
    state.hasTrack = false;
    setStatus("Add an mp3 file");
  });

  els.player.addEventListener("ended", () => {
    pauseAudio();
  });

  window.addEventListener("resize", () => {
    resizeCanvas(els.visual, visualCtx);
    resizeCanvas(els.overlay, overlayCtx);
  });
}

function init() {
  bindEvents();
  updateTrackCaption();
  drawVisual();
}

init();
