#include "accelerator_hls.h"
#include <cmath>

#include "w1.h"
#include "b1.h"
#include "w2.h"
#include "b2.h"
#include "wd1.h"
#include "bd1.h"
#include "wd2.h"
#include "bd2.h"

// ========================================================================
// 30 FPS LAYER 1
// ========================================================================
void stream_conv_maxpool_1(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    static data_t line_buffer[3][IN_W][IN_C];
    // Only partition the 3 rows. The rest is read sequentially! Saves thousands of LUTs.
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    
    // Cyclically partition weights so 32 DSPs can read from it at once
    #pragma HLS ARRAY_PARTITION variable=w1 cyclic factor=32 dim=1

    for (int h = 0; h < IN_H; h++) {
        for (int w = 0; w < IN_W; w++) {
            
            // Shift Line Buffer
            for (int c = 0; c < IN_C; c++) {
                #pragma HLS PIPELINE II=1
                data_t in_pixel = in_stream.read();
                line_buffer[0][w][c] = line_buffer[1][w][c];
                line_buffer[1][w][c] = line_buffer[2][w][c];
                line_buffer[2][w][c] = in_pixel;
            }

            if (h >= 2 && w >= 2) {
                bool pool_stride = (h % 2 == 0) && (w % 2 == 0);
                
                data_t temp_sum[CONV1_F];
                #pragma HLS ARRAY_PARTITION variable=temp_sum complete dim=1
                for (int f = 0; f < CONV1_F; f++) {
                    #pragma HLS UNROLL
                    temp_sum[f] = (data_t)b1[f];
                }
                
                // INVERTED MATH: Read 1 pixel, compute 32 filters!
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        for (int c = 0; c < IN_C; c++) {
                            #pragma HLS PIPELINE II=1
                            data_t pixel = line_buffer[kh][w - 2 + kw][c];
                            
                            for (int f = 0; f < CONV1_F; f++) {
                                #pragma HLS UNROLL
                                int w_idx = (kh * 3 * IN_C * CONV1_F) + (kw * IN_C * CONV1_F) + (c * CONV1_F) + f;
                                temp_sum[f] += pixel * (data_t)w1[w_idx];
                            }
                        }
                    }
                }
                
                if (pool_stride) {
                    for (int f = 0; f < CONV1_F; f++) {
                        #pragma HLS PIPELINE II=1
                        data_t relu_val = (temp_sum[f] > (data_t)0) ? temp_sum[f] : (data_t)0;
                        out_stream.write(relu_val);
                    }
                }
            }
        }
    }
}

// ========================================================================
// 30 FPS LAYER 2
// ========================================================================
void stream_conv_maxpool_2(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    static data_t line_buffer[3][POOL1_W][CONV1_F];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w2 cyclic factor=32 dim=1

    for (int h = 0; h < POOL1_H; h++) {
        for (int w = 0; w < POOL1_W; w++) {
            
            for (int c = 0; c < CONV1_F; c++) {
                #pragma HLS PIPELINE II=1
                data_t in_pixel = in_stream.read();
                line_buffer[0][w][c] = line_buffer[1][w][c];
                line_buffer[1][w][c] = line_buffer[2][w][c];
                line_buffer[2][w][c] = in_pixel;
            }

            if (h >= 2 && w >= 2) {
                bool pool_stride = (h % 2 == 0) && (w % 2 == 0);
                
                // Process 32 filters concurrently. Loops twice (64/32 = 2)
                for (int f_outer = 0; f_outer < CONV2_F; f_outer += 32) { 
                    
                    data_t temp_sum[32];
                    #pragma HLS ARRAY_PARTITION variable=temp_sum complete dim=1
                    
                    for (int i=0; i<32; i++) {
                        #pragma HLS UNROLL
                        temp_sum[i] = (data_t)b2[f_outer + i];
                    }
                    
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            for (int c = 0; c < CONV1_F; c++) {
                                #pragma HLS PIPELINE II=1
                                data_t pixel = line_buffer[kh][w - 2 + kw][c];
                                
                                for (int i = 0; i < 32; i++) {
                                    #pragma HLS UNROLL
                                    int f = f_outer + i;
                                    int w_idx = (kh * 3 * CONV1_F * CONV2_F) + (kw * CONV1_F * CONV2_F) + (c * CONV2_F) + f;
                                    temp_sum[i] += pixel * (data_t)w2[w_idx];
                                }
                            }
                        }
                    }
                    
                    if (pool_stride) {
                        for (int i = 0; i < 32; i++) {
                            #pragma HLS PIPELINE II=1
                            data_t relu_val = (temp_sum[i] > (data_t)0) ? temp_sum[i] : (data_t)0;
                            out_stream.write(relu_val);
                        }
                    }
                }
            }
        }
    }
}

// ========================================================================
// FAST DENSE LAYER (13ms Latency, Low LUTs)
// ========================================================================
void stream_dense(hls::stream<data_t>& in_stream, data_t out_score[1]) {
    data_t dense1_out[DENSE1_SIZE];
    #pragma HLS ARRAY_PARTITION variable=dense1_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=wd1 cyclic factor=4 dim=1
    
    for (int o = 0; o < DENSE1_SIZE; o++) {
        #pragma HLS UNROLL
        dense1_out[o] = (data_t)bd1[o];
    }

    // Unroll by a safe factor of 4. Drops latency to 13ms without overloading MUXes.
    for (int i = 0; i < (POOL2_H * POOL2_W * CONV2_F); i++) {
        data_t in_val = in_stream.read();
        for (int o_outer = 0; o_outer < DENSE1_SIZE; o_outer += 4) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                int o = o_outer + k;
                dense1_out[o] += in_val * (data_t)wd1[i * DENSE1_SIZE + o];
            }
        }
    }

    data_t final_sum = (data_t)bd2[0];
    for (int o = 0; o < DENSE1_SIZE; o++) {
        #pragma HLS PIPELINE II=1
        data_t relu_val = (dense1_out[o] > (data_t)0) ? dense1_out[o] : (data_t)0;
        final_sum += relu_val * (data_t)wd2[o * DENSE2_SIZE + 0];
    }

    out_score[0] = (data_t)(1.0f / (1.0f + std::exp((float)-final_sum)));
}

// ========================================================================
// TOP LEVEL
// ========================================================================
void myproject(hls::stream<axis_t>& input_stream, hls::stream<axis_t>& output_stream) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    #pragma HLS DATAFLOW

    hls::stream<data_t> raw_input("raw_input");
    hls::stream<data_t> pool1_out("pool1_out");
    hls::stream<data_t> pool2_out("pool2_out");

    for (int i = 0; i < IN_H * IN_W * IN_C; i++) {
        #pragma HLS PIPELINE II=1
        axis_t read_in = input_stream.read();
        raw_input.write(read_in.data);
    }

    stream_conv_maxpool_1(raw_input, pool1_out);
    stream_conv_maxpool_2(pool1_out, pool2_out);
    
    data_t final_score[1];
    stream_dense(pool2_out, final_score);

    axis_t write_out;
    write_out.data = final_score[0];
    write_out.last = true; 
    output_stream.write(write_out);
}
