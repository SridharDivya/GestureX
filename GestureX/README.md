# GestureX

Real-time hand gesture recognition that turns sign language into text and speech, with video call support and a built-in way to teach new signs.

## How it works
- **MediaPipe Hands** finds 21 landmarks per frame, in the browser (video never leaves the device for recognition).
- Landmarks are normalised (relative to wrist, scaled by hand size) and classified with **kNN**.
- A sign must be held ~12 frames to be accepted, which avoids flicker.
- **Web Speech API** reads text aloud. **WebRTC** carries the call peer-to-peer; Flask-SocketIO only relays signaling and caption text.
- **Learn mode**: type a word, record 40 frames, done. Signs are stored in the browser and can be exported/imported as JSON.

## Run
```
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000. For a call, open it in two tabs/devices, use the same room name, and press **Start call** on both. Other devices need HTTPS (e.g. `ngrok http 5000`) for camera access.

## Deploy on Render
- Root Directory: the folder that contains `app.py` (in your repo: `GestureX/GestureX`)
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --worker-class gthread --threads 100 -w 1 app:app`
- Environment: `PYTHON_VERSION=3.11.9`
- Camera needs HTTPS, which Render provides. Keep `-w 1` (room state is in memory).
- Optional, for calls on strict networks: set `TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL`.

## Desktop mode (Python + OpenCV)
```
pip install -r requirements-desktop.txt
python gesture_desktop.py collect hello
python gesture_desktop.py collect thanks
python gesture_desktop.py run
```

## Limits and next steps
- Recognises static single-hand signs. For motion signs, swap kNN for an LSTM/Transformer over landmark sequences.
- Add two-hand support (84 features) and a pretrained ASL alphabet set.
- Add TURN server for calls across strict networks.
