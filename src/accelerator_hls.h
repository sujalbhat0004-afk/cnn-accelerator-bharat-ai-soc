#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "hls_stream.h"

// 16-bit fixed point for DSP efficiency
typedef ap_fixed<16, 8> data_t;

// Your actual model dimensions
#define IN_H 150
#define IN_W 150
#define IN_C 3

#define POOL1_H 74
#define POOL1_W 74
#define CONV1_F 32

#define POOL2_H 36
#define POOL2_W 36
#define CONV2_F 64

#define DENSE1_SIZE 64
#define DENSE2_SIZE 1

// AXI-Stream structure for PYNQ DMA integration
struct axis_t {
    data_t data;
    bool last;
};

// Streaming Top Function Prototype
void myproject(hls::stream<axis_t>& input_stream, hls::stream<axis_t>& output_stream);

#endif
