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
  trackList: document.querySelector("#trackList"),
  playButton: document.querySelector("#playButton"),
  cameraButton: document.querySelector("#cameraButton"),
  fileInput: document.querySelector("#fileInput"),
  status: document.querySelector("#status"),
  player: document.querySelector("#player"),
  webcam: document.querySelector("#webcam"),
  overlay: document.querySelector("#overlay"),
  cameraPanel: document.querySelector(".camera-panel"),
  volumeReadout: document.querySelector("#volumeReadout"),
  spaceReadout: document.querySelector("#spaceReadout"),
  handReadout: document.querySelector("#handReadout"),
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
  targetVolume: 0.52,
  volume: 0.52,
  targetSpace: 0,
  space: 0,
  openness: 0,
  blobPhase: 0,
  modelReady: false,
  detectFrames: 0,
  lastStatusAt: 0,
  swipeSamples: [],
  swipeCooldownUntil: 0,
};

let audio = null;
let visionTasks = null;
let handLandmarker = null;
let webcamStream = null;
let lastVideoTime = -1;
let animationHandle = null;
let detectHandle = null;
let lastModelError = null;

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
  if (now - state.lastStatusAt < interval && els.status.textContent === text) return;

  state.lastStatusAt = now;
  setStatus(text);
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

function renderTracks() {
  els.trackList.innerHTML = "";

  state.tracks.forEach((track) => {
    const button = document.createElement("button");
    button.className = "track-button";
    button.type = "button";
    button.textContent = track.title;
    button.dataset.trackId = track.id;

    if (track.id === state.activeTrackId) {
      button.classList.add("is-active");
    }

    button.addEventListener("click", () => selectTrack(track.id));
    els.trackList.append(button);
  });
}

function selectTrack(id) {
  const track = state.tracks.find((item) => item.id === id);
  if (!track) return;

  state.activeTrackId = id;
  state.hasTrack = true;
  els.player.src = track.src;
  els.player.load();
  renderTracks();
  setStatus(track.title);

  if (state.isPlaying) {
    playAudio();
  }
}

function shiftTrack(direction) {
  if (state.tracks.length < 2) return;

  const currentIndex = state.tracks.findIndex((track) => track.id === state.activeTrackId);
  const safeIndex = currentIndex === -1 ? 0 : currentIndex;
  const nextIndex = (safeIndex + direction + state.tracks.length) % state.tracks.length;
  const nextTrack = state.tracks[nextIndex];

  selectTrack(nextTrack.id);
  setTimedStatus(nextTrack.title, 0);
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
  state.swipeSamples = [];
  state.swipeCooldownUntil = 0;
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
    state.targetSpace = lerp(state.targetSpace, 0, 0.04);
    state.swipeSamples = [];
    return;
  }

  let spaceTotal = 0;
  let opennessTotal = 0;
  let volumeTotal = 0;
  let openSwipeHand = null;

  hands.forEach((hand) => {
    const metrics = getHandMetrics(hand);

    volumeTotal += metrics.openness;
    spaceTotal += metrics.spread - metrics.cluster;
    opennessTotal += metrics.avgTipSpread;

    if (metrics.openness > 0.74 && (!openSwipeHand || metrics.openness > openSwipeHand.openness)) {
      openSwipeHand = metrics;
    }
  });

  state.handCount = hands.length;
  state.targetVolume = clamp(volumeTotal / hands.length, 0.03, 1);
  state.targetSpace = clamp(spaceTotal / hands.length, -1, 1);
  state.openness = opennessTotal / hands.length;
  updateTrackSwipe(openSwipeHand);
}

function updateTrackSwipe(openSwipeHand) {
  const now = performance.now();

  if (!openSwipeHand) {
    state.swipeSamples = [];
    return;
  }

  const displayX = 1 - openSwipeHand.center.x;
  const displayY = openSwipeHand.center.y;

  state.swipeSamples.push({ x: displayX, y: displayY, time: now });
  state.swipeSamples = state.swipeSamples.filter((sample) => now - sample.time < 520);

  if (now < state.swipeCooldownUntil || state.swipeSamples.length < 2) return;

  const first = state.swipeSamples[0];
  const last = state.swipeSamples[state.swipeSamples.length - 1];
  const dx = last.x - first.x;
  const dy = Math.abs(last.y - first.y);
  const dt = Math.max(1, last.time - first.time);
  const velocity = Math.abs(dx) / dt;

  if (Math.abs(dx) > 0.2 && dy < 0.16 && velocity > 0.00045) {
    shiftTrack(dx > 0 ? 1 : -1);
    state.swipeSamples = [];
    state.swipeCooldownUntil = now + 900;
  }
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
          setTimedStatus("Open hand controls volume, swipe to change track", 420);
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
  const baseRadius = Math.min(width, height) * 0.245;
  const spread = Math.max(0, state.space);
  const cluster = Math.max(0, -state.space);
  const pulse = 1 + (state.volume - 0.5) * 0.22;
  const lobes = 3 + Math.round(spread * 3);
  const wobble = 0.08 + spread * 0.13 + cluster * 0.04;

  state.blobPhase += 0.012 + state.volume * 0.006;
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

  els.volumeReadout.textContent = Math.round(state.volume * 100);
  els.spaceReadout.textContent = Math.round(state.space * 100);
  els.handReadout.textContent = state.handCount;
  applyAudioState();

  animationHandle = requestAnimationFrame(drawVisual);
}

function loadLocalFiles(files) {
  const newTracks = Array.from(files)
    .filter((file) => file.type.startsWith("audio/"))
    .map((file, index) => ({
      id: `local-${Date.now()}-${index}`,
      title: file.name.replace(/\.[^/.]+$/, ""),
      src: URL.createObjectURL(file),
    }));

  if (!newTracks.length) return;

  state.tracks = [...newTracks, ...state.tracks];
  state.activeTrackId = newTracks[0].id;
  renderTracks();
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

  els.fileInput.addEventListener("change", (event) => {
    loadLocalFiles(event.target.files);
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
  renderTracks();
  drawVisual();
}

init();
