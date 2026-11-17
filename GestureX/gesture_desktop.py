"""GestureX desktop version (Python + OpenCV + MediaPipe).
  python gesture_desktop.py collect hello   # record samples for a sign (press q to stop)
  python gesture_desktop.py run             # live recognition + speech
"""
import sys, csv, os, collections
import cv2, numpy as np, mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier

DATA = "signs.csv"
hands_mod, draw = mp.solutions.hands, mp.solutions.drawing_utils


def features(lm):
    pts = np.array([[p.x, p.y] for p in lm.landmark])
    pts -= pts[0]
    return (pts / (np.abs(pts).max() or 1)).flatten()


def collect(label, n=200):
    cap, count = cv2.VideoCapture(0), 0
    with hands_mod.Hands(max_num_hands=1) as h, open(DATA, "a", newline="") as f:
        w = csv.writer(f)
        while count < n:
            ok, frame = cap.read()
            if not ok: break
            r = h.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if r.multi_hand_landmarks:
                lm = r.multi_hand_landmarks[0]
                w.writerow([label, *features(lm)]); count += 1
                draw.draw_landmarks(frame, lm, hands_mod.HAND_CONNECTIONS)
            cv2.putText(frame, f"{label}: {count}/{n}", (10, 30), 0, 1, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


def run():
    import pyttsx3
    rows = list(csv.reader(open(DATA)))
    X = np.array([[float(v) for v in r[1:]] for r in rows]); y = [r[0] for r in rows]
    clf = KNeighborsClassifier(5).fit(X, y)
    tts, cap = pyttsx3.init(), cv2.VideoCapture(0)
    recent, sentence, locked = collections.deque(maxlen=12), [], False
    with hands_mod.Hands(max_num_hands=1) as h:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            r = h.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if r.multi_hand_landmarks:
                lm = r.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, lm, hands_mod.HAND_CONNECTIONS)
                v = features(lm).reshape(1, -1)
                dist, _ = clf.kneighbors(v, 1)
                recent.append(clf.predict(v)[0] if dist[0][0] < 0.6 else None)
                if len(recent) == 12 and len(set(recent)) == 1 and recent[0] and not locked:
                    sentence.append(recent[0]); locked = True
                elif len(set(recent)) > 1: locked = False
            else:
                recent.clear(); locked = False
            cv2.putText(frame, " ".join(sentence), (10, 40), 0, 1, (255, 255, 0), 2)
            cv2.imshow("GestureX  [s]=speak [c]=clear [q]=quit", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"): tts.say(" ".join(sentence)); tts.runAndWait()
            elif k == ord("c"): sentence.clear()
            elif k == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "collect": collect(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "run" and os.path.exists(DATA): run()
    else: print(__doc__)
