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
  return n > K / 2 ? label : null;
}

// ---- per-frame handler
function onResults(res) {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const lm = res.multiHandLandmarks && res.multiHandLandmarks[0];
  if (!lm) { candidate = null; held = 0; locked = false; $("live").textContent = "Show a sign"; return; }
  drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: "#7ea2ff", lineWidth: 3 });
  drawLandmarks(ctx, lm, { color: "#fff", radius: 3 });
  const vec = features(lm);

  if (recording) {
    samples.push({ label: recording.label, vec });
    $("live").textContent = `Recording "${recording.label}" ${40 - recording.left + 1}/40`;
    if (--recording.left <= 0) { recording = null; save(); status("Saved. Try the sign now."); $("recBtn").disabled = false; }
    return;
  }
  const label = classify(vec);
  $("live").textContent = label || "Unknown sign";
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
  recording = { label, left: 40 }; $("recBtn").disabled = true;
};
$("exportBtn").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(samples)], { type: "application/json" }));
  a.download = "gesturex_signs.json"; a.click();
};
$("importBtn").onclick = () => $("file").click();
$("file").onchange = async e => {
  try { const d = JSON.parse(await e.target.files[0].text()); samples = samples.concat(d); save(); status("Imported."); }
  catch { status("That file isn't a valid signs export."); }
};
$("resetBtn").onclick = () => { if (confirm("Delete all taught signs?")) { samples = []; save(); } };
$("speakBtn").onclick = () => speak(words.join(" "));
$("clearBtn").onclick = () => { words = []; $("text").textContent = ""; };
$("sendBtn").onclick = () => words.length && socket.emit("caption", { text: words.join(" ") });

// ---- camera + MediaPipe
const hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}` });
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: .6, minTrackingConfidence: .6 });
hands.onResults(onResults);

let stream;
(async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true });
    video.srcObject = stream; await video.play();
    const loop = async () => { if (video.readyState >= 2) await hands.send({ image: video }); requestAnimationFrame(loop); };
    loop();
  } catch { status("Camera blocked. Allow camera access (works on localhost or HTTPS)."); }
})();
renderSigns();

// ---- WebRTC call (video peer-to-peer; only signaling and captions touch the server)
const socket = io();
let pc;
function makePC() {
  pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
  stream.getTracks().forEach(t => pc.addTrack(t, stream));
  pc.ontrack = e => { const r = $("remote"); r.srcObject = e.streams[0]; r.style.display = "block"; };
  pc.onicecandidate = e => e.candidate && socket.emit("signal", { data: { ice: e.candidate } });
}
$("callBtn").onclick = () => { if (!stream) return status("Start the camera first."); socket.emit("join", { room: $("room").value || "lobby" }); status("Waiting for the other person to join this room..."); };
socket.on("peer-joined", async () => {
  makePC(); const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
  socket.emit("signal", { data: { sdp: pc.localDescription } });
});
socket.on("signal", async d => {
  if (d.sdp) {
    if (!pc) makePC();
    await pc.setRemoteDescription(d.sdp);
    if (d.sdp.type === "offer") { const a = await pc.createAnswer(); await pc.setLocalDescription(a); socket.emit("signal", { data: { sdp: pc.localDescription } }); }
    status("Call connected.");
  } else if (d.ice && pc) pc.addIceCandidate(d.ice).catch(() => {});
});
socket.on("caption", t => { const el = $("remoteText"); el.style.display = "block"; el.textContent = t; speak(t); });
function hang() { if (pc) pc.close(); pc = null; $("remote").style.display = "none"; $("remoteText").style.display = "none"; }
socket.on("peer-left", hang);
$("hangBtn").onclick = () => { hang(); socket.disconnect(); socket.connect(); status("Call ended."); };
