import cv2, numpy as np, time, os
import urllib.request
from IPython.display import display, Image, clear_output
from pynq import Overlay, Xlnk, MMIO

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
DMA_ADDRESS = 0x41E00000 
RANGE       = 65536

MM2S_DMACR  = 0x00; MM2S_SR = 0x04; MM2S_SA = 0x18; MM2S_LENGTH = 0x28
S2MM_DMACR  = 0x30; S2MM_SR = 0x34; S2MM_DA = 0x48; S2MM_LENGTH = 0x58

print("Loading Bitstream...")
overlay = Overlay("tinymask.bit")

for ip in overlay.ip_dict:
    if 'interrupts' in overlay.ip_dict[ip]:
        overlay.ip_dict[ip]['interrupts'] = {}

ai_ip = overlay.myproject_0

dma_mmio = MMIO(DMA_ADDRESS, RANGE)
xlnk = Xlnk()

in_buf  = xlnk.cma_array(shape=(67500,), dtype=np.int16)
out_buf = xlnk.cma_array(shape=(2,), dtype=np.int16)
fast_cpu_buffer = np.empty((67500,), dtype=np.int16)

print("✅ Hardware Ready.")

# ==========================================
# 2. IRONCLAD HARDWARE HANDSHAKE
# ==========================================
def run_kernel_sync():
    """Strictly ordered DMA triggers to prevent Race Conditions."""
    
    # 1. Ensure IP is halted before we do anything
    ai_ip.write(0x00, 0x00)
    
    # 2. Set memory addresses
    dma_mmio.write(S2MM_DA, out_buf.physical_address)
    dma_mmio.write(MM2S_SA, in_buf.physical_address)
    
    # 3. Start DMA Channels (Listening mode ON)
    dma_mmio.write(S2MM_DMACR, 1)
    dma_mmio.write(MM2S_DMACR, 1)
    
    # 4. Set Lengths (S2MM FIRST so it is ready to catch the output)
    dma_mmio.write(S2MM_LENGTH, 4)       
    dma_mmio.write(MM2S_LENGTH, 135000)  
    
    # 5. NOW fire the AI IP! (Data flows perfectly)
    ai_ip.write(0x00, 0x01)
    
    # 6. Wait for completion
    timeout = 20000
    while not (dma_mmio.read(S2MM_SR) & 0x02) and timeout > 0:
        timeout -= 1
        
    timeout = 20000
    while not (dma_mmio.read(MM2S_SR) & 0x02) and timeout > 0:
        timeout -= 1

# ==========================================
# 3. MAIN LOOP
# ==========================================
# SWITCHED BACK TO ALT2 FOR BETTER MASK TRACKING
xml_path = 'haarcascade_frontalface_alt2.xml'
if not os.path.exists(xml_path):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml', xml_path)

face_cascade = cv2.CascadeClassifier(xml_path)
cap = cv2.VideoCapture(0)
cap.set(3, 320); cap.set(4, 240) 

THRESHOLD = 0.50 

try:
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_tiny = cv2.resize(gray, (160, 120))
        # More aggressive face tracking parameters so it doesn't lose you with the mask on
        faces = face_cascade.detectMultiScale(gray_tiny, 1.1, 2, minSize=(20, 20))

        inf_latency = 0
        score = 0.0

        for (x_t, y_t, w_t, h_t) in faces:
            x, y, w, h = x_t*2, y_t*2, w_t*2, h_t*2
            
            # Expand the bounding box slightly so the AI sees the edges of the mask
            y_exp = max(0, y - 10); h_exp = min(240 - y_exp, h + 20)
            x_exp = max(0, x - 10); w_exp = min(320 - x_exp, w + 20)
            
            roi_frame = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
            if roi_frame.size == 0: continue
            
            # Format Data
            roi_150 = cv2.resize(roi_frame, (150, 150))
            # Reverse BGR to RGB so colors map correctly in FPGA
            roi_150 = cv2.cvtColor(roi_150, cv2.COLOR_BGR2RGB)
            
            fast_cpu_buffer[:] = roi_150.flatten()
            np.copyto(in_buf, fast_cpu_buffer)
            
            # Run Synchronized FPGA Inference
            t_inf = time.time()
            run_kernel_sync()
            inf_latency = (time.time() - t_inf) * 1000
            
            # Calculate Score
            raw_hw_score = out_buf[0]
            score = float(raw_hw_score) / 256.0
            
            if score > THRESHOLD:
                label, color = "NO MASK", (0, 0, 255)
            else:
                label, color = "MASK", (0, 255, 0)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = 1.0 / (time.time() - t_start)
        
        cv2.rectangle(frame, (0, 0), (180, 55), (0,0,0), -1)
        cv2.putText(frame, f"FPGA Time: {inf_latency:.1f}ms", (10, 20), 0, 0.45, (255,255,255), 1)
        cv2.putText(frame, f"System FPS: {fps:.1f}", (10, 45), 0, 0.45, (0,255,255), 1)

        _, jpeg = cv2.imencode('.jpg', frame)
        clear_output(wait=True)
        display(Image(data=jpeg.tobytes()))

except KeyboardInterrupt:
    print("Stopped.")
finally:
    cap.release()
    xlnk.xlnk_reset()
