// GestureX: in-browser hand tracking (MediaPipe) + kNN sign classifier + speech + WebRTC call.
const $ = id => document.getElementById(id);
const video = $("local"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const K = 5, MAX_DIST = 0.6, HOLD_FRAMES = 12, STORE = "gesturex_samples";

let samples = []; try { samples = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch {}
let recording = null;            // {label, left}
let candidate = null, held = 0, locked = false;
let words = [];

const save = () => { try { localStorage.setItem(STORE, JSON.stringify(samples)); } catch {} renderSigns(); };
const status = t => $("status").textContent = t;
const live = t => $("live").textContent = t;
const fail = t => { live(t); status(t); };

// ---- features: 21 landmarks -> 42 numbers, relative to wrist, scaled to hand size
function features(lm) {
  const w = lm[0];
  const pts = lm.map(p => [p.x - w.x, p.y - w.y]);
  const s = Math.max(...pts.map(p => Math.hypot(p[0], p[1]))) || 1;
  return pts.flatMap(p => [p[0] / s, p[1] / s]);
}
const dist = (a, b) => Math.sqrt(a.reduce((t, v, i) => t + (v - b[i]) ** 2, 0));

function classify(vec) {
  if (!samples.length) return null;
  const near = samples.map(s => ({ l: s.label, d: dist(vec, s.vec) }))
    .sort((a, b) => a.d - b.d).slice(0, K);
  if (near[0].d > MAX_DIST) return null;
  const votes = {};
  near.forEach(n => votes[n.l] = (votes[n.l] || 0) + 1);
  const [label, n] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  return n > Math.min(K, near.length) / 2 ? label : null;
}

// ---- per-frame handler
let modelReady = false;
function onResults(res) {
  modelReady = true;
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const lm = res.multiHandLandmarks && res.multiHandLandmarks[0];
  if (!lm) { candidate = null; held = 0; locked = false; live("Show a sign"); return; }
  drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: "#7ea2ff", lineWidth: 3 });
  drawLandmarks(ctx, lm, { color: "#fff", radius: 3 });
  const vec = features(lm);

  if (recording) {
    samples.push({ label: recording.label, vec });
    live(`Recording "${recording.label}" ${40 - recording.left + 1}/40`);
    if (--recording.left <= 0) { recording = null; save(); status("Saved. Try the sign now."); $("recBtn").disabled = false; }
    return;
  }
  const label = classify(vec);
  live(label || "Unknown sign");
  if (label && label === candidate) held++; else { candidate = label; held = 0; locked = false; }
  if (label && held >= HOLD_FRAMES && !locked) { locked = true; addWord(label); }
}

function addWord(w) {
  words.push(w); $("text").textContent = words.join(" ");
  if ($("autoSpeak").checked) speak(w);
}
function speak(t) {
  if (!t || !window.speechSynthesis) return status("Speech is not supported in this browser.");
  speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(t));
}

// ---- learn mode
function renderSigns() {
  const counts = {}; samples.forEach(s => counts[s.label] = (counts[s.label] || 0) + 1);
  $("signs").innerHTML = Object.keys(counts).length
    ? Object.entries(counts).map(([l, n]) => `<b>${l.replace(/</g, "&lt;")} (${n})</b>`).join("")
    : '<span class="mute">No signs yet. Teach your first one above.</span>';
}
$("recBtn").onclick = () => {
  const label = $("label").value.trim();
  if (!label) return status("Type the word for this sign first.");
  if (!modelReady) return status("Hand tracking is not running yet. Start the camera first.");
  recording = { label, left: 40 }; $("recBtn").disabled = true;
  status("Hold the sign steady in front of the camera...");
};
$("exportBtn").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(samples)], { type: "application/json" }));
  a.download = "gesturex_signs.json"; a.click();
};
$("importBtn").onclick = () => $("file").click();
$("file").onchange = async e => {
  try {
    const d = JSON.parse(await e.target.files[0].text());
    if (!Array.isArray(d) || d.some(s => !s.label || !Array.isArray(s.vec))) throw 0;
    samples = samples.concat(d); save(); status("Imported.");
  } catch { status("That file isn't a valid signs export."); }
  e.target.value = "";
};
$("resetBtn").onclick = () => { if (confirm("Delete all taught signs?")) { samples = []; save(); } };
$("speakBtn").onclick = () => speak(words.join(" "));
$("clearBtn").onclick = () => { words = []; $("text").textContent = ""; };
$("sendBtn").onclick = () => {
  if (!words.length) return status("Nothing to send yet.");
  socket.emit("caption", { text: words.join(" ") });
  status("Sent to caller.");
};

// ---- camera + MediaPipe
let stream, hands, loopRunning = false;

async function getStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    throw Object.assign(new Error("no-api"), { name: "NoApi" });
  const video_ = { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" };
  try {
    return await navigator.mediaDevices.getUserMedia({ video: video_, audio: true });
  } catch (e) {
    // No microphone (or mic blocked)? Still run with video only.
    return await navigator.mediaDevices.getUserMedia({ video: video_, audio: false });
  }
}

function cameraError(e) {
  const n = e && e.name;
  if (n === "NotAllowedError" || n === "SecurityError") return "Camera blocked. Click the camera icon in the address bar, choose Allow, then press Start camera.";
  if (n === "NotFoundError" || n === "OverconstrainedError") return "No camera found. Plug one in, then press Start camera.";
  if (n === "NotReadableError" || n === "AbortError") return "Camera is in use by another app or tab. Close it, then press Start camera.";
  if (n === "NoApi") return "Camera needs HTTPS (or localhost). Open the https:// address of this site.";
  return "Could not start the camera: " + (n || e);
}

async function startCamera() {
  $("camBtn").disabled = true;
  live("Starting camera...");
  try {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await getStream();
    video.srcObject = stream;
    await video.play();
  } catch (e) { $("camBtn").disabled = false; return fail(cameraError(e)); }
  $("camBtn").style.display = "none";
  status("Camera on. Loading hand tracking...");
  live("Loading hand tracking...");
  await startTracking();
}

async function startTracking() {
  if (typeof Hands === "undefined" || typeof drawConnectors === "undefined")
    return fail("Could not load the hand-tracking library (internet or ad-blocker?). Reload the page.");
  try {
    if (!hands) {
      hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}` });
      hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: .6, minTrackingConfidence: .6 });
      hands.onResults(onResults);
      await hands.initialize();
    }
  } catch (e) { hands = null; return fail("Hand-tracking model failed to load. Reload the page and try again."); }
  status("Ready. Show a sign.");
  live("Show a sign");
  if (loopRunning) return;
  loopRunning = true;
  const loop = async () => {
    try {
      if (video.readyState >= 2 && video.videoWidth) await hands.send({ image: video });
    } catch (e) { /* skip a bad frame */ }
    requestAnimationFrame(loop);
  };
  loop();
}
$("camBtn").onclick = startCamera;
renderSigns();
startCamera();

// ---- WebRTC call (video peer-to-peer; only signaling and captions touch the server)
const socket = io();
let pc, iceServers = [{ urls: "stun:stun.l.google.com:19302" }], pendingIce = [];
fetch("/ice").then(r => r.json()).then(s => { if (Array.isArray(s)) iceServers = s; }).catch(() => {});

function makePC() {
  if (pc) pc.close();
  pendingIce = [];
  pc = new RTCPeerConnection({ iceServers });
  stream.getTracks().forEach(t => pc.addTrack(t, stream));
  pc.ontrack = e => {
    const r = $("remote"); r.srcObject = e.streams[0]; r.style.display = "block";
    r.play().catch(() => {});
  };
  pc.onicecandidate = e => e.candidate && socket.emit("signal", { data: { ice: e.candidate } });
  pc.onconnectionstatechange = () => {
    if (!pc) return;
    if (pc.connectionState === "connected") status("Call connected.");
    else if (pc.connectionState === "failed") status("Call could not connect (strict network). Add a TURN server, see README.");
  };
}
async function flushIce() {
  for (const c of pendingIce.splice(0)) await pc.addIceCandidate(c).catch(() => {});
}

$("callBtn").onclick = () => {
  if (!stream) return status("Start the camera first.");
  socket.emit("join", { room: $("room").value.trim() || "lobby" });
  status("Waiting for the other person to join this room...");
};
socket.on("room-full", () => status("That room already has two people. Pick another room name."));
socket.on("peer-joined", async () => {
  try {
    makePC();
    const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
    socket.emit("signal", { data: { sdp: pc.localDescription } });
    status("Connecting...");
  } catch (e) { status("Could not start the call."); }
});

// handle signals one at a time so ICE never races the offer/answer
let chain = Promise.resolve();
socket.on("signal", d => { chain = chain.then(() => handleSignal(d)).catch(() => {}); });
async function handleSignal(d) {
  if (d.sdp) {
    if (d.sdp.type === "offer") { if (!stream) return; makePC(); }
    if (!pc) return;
    await pc.setRemoteDescription(d.sdp);
    await flushIce();
    if (d.sdp.type === "offer") {
      const a = await pc.createAnswer(); await pc.setLocalDescription(a);
      socket.emit("signal", { data: { sdp: pc.localDescription } });
    }
  } else if (d.ice) {
    if (pc && pc.remoteDescription) await pc.addIceCandidate(d.ice).catch(() => {});
    else pendingIce.push(d.ice);
  }
}
socket.on("caption", t => { const el = $("remoteText"); el.style.display = "block"; el.textContent = t; speak(t); });
function hang() { if (pc) pc.close(); pc = null; pendingIce = []; const r = $("remote"); r.srcObject = null; r.style.display = "none"; $("remoteText").style.display = "none"; }
socket.on("peer-left", () => { hang(); status("The other person left."); });
$("hangBtn").onclick = () => { hang(); socket.disconnect(); socket.connect(); status("Call ended."); };
