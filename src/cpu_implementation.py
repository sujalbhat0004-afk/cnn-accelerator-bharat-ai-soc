import cv2
import numpy as np
import time
from IPython.display import display, Image, clear_output

# ==============================================================================
# 1. THE AI ENGINE
# ==============================================================================
class TinyMaskNet_CPU:
    def __init__(self):
        try:
            self.w1 = np.float32(np.load("w1.npy"))
            self.b1 = np.float32(np.load("b1.npy"))
            self.w2 = np.float32(np.load("w2.npy"))
            self.b2 = np.float32(np.load("b2.npy"))
            self.wd1 = np.float32(np.load("wd1.npy"))
            self.bd1 = np.float32(np.load("bd1.npy"))
            self.wd2 = np.float32(np.load("wd2.npy"))
            self.bd2 = np.float32(np.load("bd2.npy"))
            print("✅ Weights Loaded.")
        except Exception as e:
            print(f"❌ Load Error: {e}")

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        x = np.clip(x, -15, 15)
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, img_64x64):
        # 1. Split the image into 3 separate channels (R, G, B)
        img_32 = np.float32(img_64x64)
        ch0 = img_32[:,:,0]
        ch1 = img_32[:,:,1]
        ch2 = img_32[:,:,2]
        
        # --- Layer 1: Conv2D (8 filters) ---
        l1_out = np.zeros((64, 64, 8), dtype=np.float32)
        for f in range(8):
            res0 = cv2.filter2D(ch0, -1, np.float32(self.w1[:,:,0,f]))
            res1 = cv2.filter2D(ch1, -1, np.float32(self.w1[:,:,1,f]))
            res2 = cv2.filter2D(ch2, -1, np.float32(self.w1[:,:,2,f]))
            l1_out[:,:,f] = self.relu(res0 + res1 + res2 + np.float32(self.b1[f]))
        
        # --- Layer 2: MaxPooling (Resize 64->32) ---
        x = cv2.resize(l1_out, (32, 32), interpolation=cv2.INTER_NEAREST)
        
        # --- Layer 3: Conv2D (16 filters) ---
        l2_out = np.zeros((32, 32, 16), dtype=np.float32)
        for f in range(16):
            sum_val = np.zeros((32, 32), dtype=np.float32)
            for c in range(8):
                kernel = np.float32(self.w2[:,:,c,f])
                sum_val += cv2.filter2D(x[:,:,c], -1, kernel)
            l2_out[:,:,f] = self.relu(sum_val + np.float32(self.b2[f]))
            
        # --- Layer 4: MaxPooling (Resize 32->16) ---
        x_pool = cv2.resize(l2_out, (16, 16), interpolation=cv2.INTER_NEAREST)

        # --- Layer 5: Global Average Pooling ---
        x_gap = np.mean(x_pool, axis=(0, 1), dtype=np.float32) 

        # --- Layer 6: Dense ---
        x_dense = self.relu(np.dot(x_gap, self.wd1) + self.bd1)
        score = self.sigmoid(np.dot(x_dense, self.wd2) + self.bd2)
        
        prob = float(score[0])
        return np.array([prob, 1.0 - prob])

# ==============================================================================
# 2. MAIN LOOP WITH ON-SCREEN DEBUGGING & METRICS
# ==============================================================================
ai = TinyMaskNet_CPU()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

print("🚀 SYSTEM STARTING... Watch for RED text if an error occurs.")

try:
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        
        inf_latency = 0.0
        
        for (x, y, w, h) in faces:
            # Placeholder Blue Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            try:
                roi = cv2.resize(rgb_frame[y:y+h, x:x+w], (64, 64))
                roi_norm = np.float32(roi / 255.0)
                
                # --- START LATENCY TIMER ---
                t_inf = time.time()
                scores = ai.predict(roi_norm)
                inf_latency = (time.time() - t_inf) * 1000
                # --- END LATENCY TIMER ---
                
                class_id = np.argmax(scores)
                conf = scores[class_id]
                
                label, color = ("MASK", (0, 255, 0)) if class_id == 1 else ("NO MASK", (0, 0, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} {int(conf*100)}%", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
            except Exception as e:
                # ON-SCREEN DEBUGGER
                error_msg = f"ERR: {str(e)[:40]}" 
                cv2.rectangle(frame, (10, 200), (310, 230), (0, 0, 0), -1)
                cv2.putText(frame, error_msg, (15, 220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # --- FPS & METRICS UI ---
        fps = 1.0 / (time.time() - t_start)
        cv2.rectangle(frame, (0, 0), (180, 55), (0,0,0), -1)
        cv2.putText(frame, f"CPU Time: {inf_latency:.1f}ms", (10, 20), 0, 0.45, (255,255,255), 1)
        cv2.putText(frame, f"System FPS: {fps:.1f}", (10, 45), 0, 0.45, (0,255,255), 1)

        # Output to Jupyter
        _, jpeg = cv2.imencode('.jpg', frame)
        clear_output(wait=True)
        display(Image(data=jpeg.tobytes()))
        
except KeyboardInterrupt:
    cap.release()
    print("Stopped.")
