open_project -reset out.prj
set_top gemm_fp32
add_files kernel.cpp
add_files -blackbox [file normalize "../blackbox/hf_add_f32.json"]
add_files -blackbox [file normalize "../blackbox/hf_mul_f32.json"]
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10
config_compile -pipeline_loops 0
csynth_design
exit
