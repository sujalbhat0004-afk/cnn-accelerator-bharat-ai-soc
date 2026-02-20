# Project Report

**Project Title:** Custom CNN Model for Mask Detection with Hardware Acceleration  
**Institution:** Tezpur University  
**Target Platform:** Xilinx Zynq-7000 SoC (ZedBoard xc7z020-clg484-1)  

---

![block_design](./block_design.jpeg)

Demo Link : [here](https://drive.google.com/file/d/13moP2f3kCbjELr_51sMtAtKYN_kvNh7c/view?usp=drivesdk)

## 1. Hardware and Software Components
To develop, synthesize, and deploy this project, we utilized the following stack:



* **Hardware Setup:** Xilinx Zynq-7000 SoC (ZedBoard), Logitech 720p USB Web Camera, Ethernet Cable (for network routing).
* **Software & Toolchain:** PYNQ OS v2.4 (Embedded Linux), Jupyter Notebook (OS control and execution), PuTTY (Serial communication), Xilinx Vitis HLS (C++ to RTL synthesis), and Xilinx Vivado (Hardware integration and bitstream generation).

---

## 2. Methodology & Design Partitioning
The primary objective of this project was to implement a real-time, edge-AI face mask detection system by balancing convolutional neural network (CNN) accuracy with the strict resource constraints of an embedded System-on-Chip (SoC). 



We employed a strict **Hardware/Software Co-design Strategy** to partition the workload:
* **Processing System (PS - ARM Cortex-A9):** Handles control-heavy, sequential tasks. It is responsible for video capture, dynamic Region of Interest (ROI) extraction using OpenCV Haar Cascades, and orchestrating AXI-DMA memory transfers.
* **Programmable Logic (PL - FPGA Fabric):** Handles highly parallelizable mathematical tasks. It acts as a custom hardware accelerator to execute the heavy multiply-accumulate (MAC) computations of the CNN.

**AI Architecture:**
A custom, lightweight 3-layer CNN was developed to distinguish mask fabric from facial features and shadows.
* **Input Shape:** 150 x 150 x 3 (RGB)
* **Feature Extraction:** 3 cascaded Conv2D layers (16, 32, and 64 filters) with MaxPooling.
* **Classification:** Global Average Pooling leading to a Dense classifier.

---

## 3. Hardware Utilization
The CNN algorithm was synthesized into Register-Transfer Level (RTL) hardware using Xilinx Vitis HLS. The implementation is highly optimized, maximizing the logic capabilities of the Zynq-7020 chip without exceeding its limits.



**Vivado HLS Resource Estimates:**

| Hardware Resource | Used | Available | Utilization |
| :--- | :--- | :--- | :--- |
| **LUT (Look-Up Tables)** | 46,874 | 53,200 | 88% |
| **DSP48E Slices** | 78 | 220 | 35% |
| **FF (Flip-Flops)** | 26,131 | 106,400 | 24% |
| **BRAM_18K (Block RAM)** | 51 | 280 | 18% |

---

## 4. Optimization Techniques
To transition from a software-bound bottleneck to real-time hardware execution, several critical optimization techniques were applied:

* **Fixed-Point Quantization (`ap_fixed<16, 8>`):** Floating-point arithmetic was completely eliminated in the PL. All model weights and activations were quantized to 16-bit fixed-point integers. By allocating 8 bits to the fractional component, the system maps raw 8-bit image pixels (0-255) directly into hardware without requiring expensive floating-point normalization on the CPU.
* **Direct Memory Access (DMA) Handshaking:** A race condition between the AXI-DMA and the AI IP was eliminated using explicit software-to-hardware handshaking. Auto-restart was disabled, and the IP execution (`0x01`) is triggered only after the memory buffers are primed.
* **High-Speed Memory Casting:** Python system-call overhead was bypassed by formatting frame data in CPU RAM and bursting it to the Contiguous Memory Allocator (CMA) via `np.copyto()`. 
* **HLS Pipelining:** Dataflow directives and loop unrolling were implemented in C++ to allow the 78 active DSP slices to process multiple convolutional filters simultaneously.

---

## 5. Performance Analysis & Comparison
The hardware acceleration provided a massive improvement in system throughput and inference latency, proving the viability of the Zynq-7000 for edge-AI applications.

**System Comparison:**

| Performance Metric | CPU-Only Implementation (ARM PS) | Hardware-Accelerated (FPGA PL) |
| :--- | :--- | :--- |
| **Inference Latency** | ~380ms to 450ms | **~26.0ms** (15x Reduction) |
| **System Throughput** | ~2.2 FPS | **12.0 – 15.0 FPS** (6x Boost) |
| **Resource Usage** | 100% CPU Load (Causing bottlenecks) | **88% LUT, 35% DSP, 18% BRAM** |
| **Power Profile** | High (Prolonged 100% CPU cycles) | **Moderate** (Efficient PL offloading) |
| **Model Accuracy** | ~85% (Float32 Precision) | **~70% - 80% (Int16 Precision)** |

**Accuracy Trade-off Analysis:**
A drop in classification accuracy was observed between the software and hardware implementations. Due to the strict resource limitations of the ZedBoard (specifically the ceiling on available DSP slices and BRAM) and it being the only compatible board we had access to, we could not implement a comparatively large and highly accurate CNN model. Consequently, we were forced to compromise on the model's depth and rely heavily on 16-bit fixed-point quantization. This resulted in a deliberate but necessary compromise in raw accuracy to successfully achieve real-time (15 FPS) hardware-accelerated performance on this specific embedded platform.

**Conclusion:**
By successfully migrating the convolutional workload from the ARM PS to the FPGA PL, the project achieved a 15x acceleration in mathematical execution. The resulting system is a robust, real-time edge device capable of effective mask detection.
