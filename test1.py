import cv2
import math
import subprocess
import mediapipe as mp

def get_current_volume():
    cmd = ['osascript', '-e', 'output volume of (get volume settings)']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())

def set_volume(volume_level):
    volume_level = max(0, min(100, volume_level))
    subprocess.run(['osascript', '-e', f'set volume output volume {volume_level}'])
    print(f"Volume: {volume_level}%")

class VolumeController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.min_dist = 20
        self.max_dist = 200
        self.curr_vol = get_current_volume()
        self.prev_dist = None
        self.smooth_factor = 0.3
        
        self.THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
        self.INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        pinch_dist = None
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            
            self.mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            thumb = hand.landmark[self.THUMB_TIP]
            index = hand.landmark[self.INDEX_TIP]
            
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            
            pinch_dist = math.hypot(tx - ix, ty - iy)
            
            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 2)
            cv2.circle(frame, (tx, ty), 8, (255, 0, 0), -1)
            cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
            
            cv2.putText(frame, f"{int(pinch_dist)}", (tx-20, ty-20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame, pinch_dist
    
    def update_volume(self, distance):
        if distance is None:
            return
        
        distance = max(self.min_dist, min(self.max_dist, distance))
        volume_pct = ((distance - self.min_dist) / (self.max_dist - self.min_dist)) * 100
        
        if self.prev_dist:
            volume_pct = self.smooth_factor * volume_pct + (1 - self.smooth_factor) * self.curr_vol
        
        new_vol = int(volume_pct)
        if abs(new_vol - self.curr_vol) >= 2:
            set_volume(new_vol)
            self.curr_vol = new_vol
        
        self.prev_dist = distance
    
    def run(self):
        print("Volume Control: Pinch IN = lower, OUT = higher")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            frame, distance = self.process_frame(frame)
            
            self.update_volume(distance)
            
            cv2.putText(frame, f"Volume: {self.curr_vol}%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Volume Control", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        VolumeController().run()
    except Exception as e:
        print(f"Error: {e}") 