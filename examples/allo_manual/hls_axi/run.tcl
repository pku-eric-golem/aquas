# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function of the design is "gemm_fp32"
set_top gemm_fp32

# Add design and testbench files
add_files kernel.cpp
add_files -blackbox [file normalize "../blackbox/hf_add_f32.json"]
add_files -blackbox [file normalize "../blackbox/hf_mul_f32.json"]
add_files -tb host.cpp -cflags "-std=gnu++0x"
open_solution "solution1"

# Target device is pynqz2
set_part {xc7z020clg400-1}

# Target frequency
create_clock -period 10.00

# Run HLS
csynth_design

exit
