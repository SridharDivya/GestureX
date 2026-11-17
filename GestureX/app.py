"""GestureX server: serves the web app and relays WebRTC signaling + captions.
Video and hand tracking stay in the browser; only signaling/text pass through here."""
import os
from flask import Flask, request, send_from_directory, jsonify
from flask_socketio import SocketIO, join_room, leave_room, emit

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
# "threading" matches gunicorn's gthread worker (needs simple-websocket for WebSockets).
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
members = {}  # sid -> room


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    return "ok"


@app.route("/ice")
def ice():
    """ICE servers for WebRTC. Add TURN_URL / TURN_USERNAME / TURN_CREDENTIAL env vars
    on Render to make calls work across strict networks (mobile data, offices)."""
    servers = [{"urls": "stun:stun.l.google.com:19302"}, {"urls": "stun:stun1.l.google.com:19302"}]
    turn = os.environ.get("TURN_URL")
    if turn:
        servers.append({
            "urls": [u.strip() for u in turn.split(",")],
            "username": os.environ.get("TURN_USERNAME", ""),
            "credential": os.environ.get("TURN_CREDENTIAL", ""),
        })
    return jsonify(servers)


@socketio.on("join")
def on_join(data):
    room = str((data or {}).get("room", "lobby")).strip()[:40] or "lobby"
    old = members.get(request.sid)
    if old == room:
        return
    if old:
        leave_room(old)
        emit("peer-left", {}, to=old)
    if sum(1 for r in members.values() if r == room) >= 2:
        emit("room-full", {})
        return
    members[request.sid] = room
    join_room(room)
    emit("joined", {"room": room})
    emit("peer-joined", {}, to=room, include_self=False)  # existing peer sends the offer


@socketio.on("signal")
def on_signal(data):
    room = members.get(request.sid)
    if room:
        emit("signal", data["data"], to=room, include_self=False)


@socketio.on("caption")
def on_caption(data):
    room = members.get(request.sid)
    if room:
        emit("caption", str(data["text"])[:500], to=room, include_self=False)


@socketio.on("disconnect")
def on_disconnect():
    room = members.pop(request.sid, None)
    if room:
        leave_room(room)
        emit("peer-left", {}, to=room)


if __name__ == "__main__":
    # getUserMedia works on localhost; use HTTPS (e.g. ngrok) for other devices.
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
