"""GestureX server: serves the web app and relays WebRTC signaling + captions.
Video and hand tracking stay in the browser; only signaling/text pass through here."""
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, join_room, leave_room, emit

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")
members = {}  # sid -> room


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@socketio.on("join")
def on_join(data):
    room = str(data.get("room", "lobby"))[:40]
    members[request.sid] = room
    join_room(room)
    emit("peer-joined", {}, to=room, include_self=False)  # existing peer sends the offer


@socketio.on("signal")
def on_signal(data):
    emit("signal", data["data"], to=members.get(request.sid), include_self=False)


@socketio.on("caption")
def on_caption(data):
    emit("caption", data["text"], to=members.get(request.sid), include_self=False)


@socketio.on("disconnect")
def on_disconnect():
    room = members.pop(request.sid, None)
    if room:
        leave_room(room)
        emit("peer-left", {}, to=room)


if __name__ == "__main__":
    # getUserMedia works on localhost; use HTTPS (e.g. ngrok) for other devices.
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
