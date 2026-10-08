#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @gemm_fp32(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: i8, %arg4: memref<64xf32>) attributes {itypes = "___u_", otypes = ""} {
    %alloc = memref.alloc() {name = "acc"} : memref<8x8xf32, #map>
    %cst = arith.constant {name = "cst"} 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%alloc : memref<8x8xf32, #map>)
    affine.for %arg5 = 0 to 8 {
      %alloc_0 = memref.alloc() : memref<8xf32>
      affine.for %arg6 = 0 to 8 {
        affine.store %cst, %alloc_0[%arg6] : memref<8xf32>
      } {buffer, loop_name = "j_init", pipeline_ii = 1 : i32}
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 8 {
          %0 = affine.load %arg0[%arg5 * 8 + %arg6] {from = "A"} : memref<64xf32>
          %1 = affine.load %arg1[%arg7 * 8 + %arg6] {from = "BT"} : memref<64xf32>
          %2 = arith.mulf %0, %1 : f32
          %alloc_1 = memref.alloc() {name = "product"} : memref<f32>
          affine.store %2, %alloc_1[] {to = "product"} : memref<f32>
          %3 = affine.load %alloc_0[%arg7] : memref<8xf32>
          %4 = affine.load %alloc_1[] {from = "product"} : memref<f32>
          %5 = arith.addf %3, %4 : f32
          affine.store %5, %alloc_0[%arg7] : memref<8xf32>
        } {loop_name = "j", pipeline_ii = 1 : ui32}
      } {loop_name = "k", op_name = "S_k_0", reduction}
      affine.for %arg6 = 0 to 8 {
        %0 = affine.load %alloc_0[%arg6] : memref<8xf32>
        affine.store %0, %alloc[%arg5, %arg6] : memref<8x8xf32, #map>
      } {buffer, loop_name = "j_back", pipeline_ii = 1 : i32}
    } {loop_name = "i", op_name = "mac"}
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 8 {
        %alloc_0 = memref.alloc() {name = "initial"} : memref<f32>
        affine.store %cst, %alloc_0[] {to = "initial"} : memref<f32>
        %0 = arith.extui %arg3 : i8 to i32
        %c0_i32 = arith.constant 0 : i32
        %1 = arith.cmpi ne, %0, %c0_i32 : i32
        scf.if %1 {
          %5 = affine.load %arg2[%arg5 * 8 + %arg6] {from = "C"} : memref<64xf32>
          affine.store %5, %alloc_0[] {to = "initial"} : memref<f32>
        }
        %2 = affine.load %alloc[%arg5, %arg6] {from = "acc"} : memref<8x8xf32, #map>
        %3 = affine.load %alloc_0[] {from = "initial"} : memref<f32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %arg4[%arg5 * 8 + %arg6] {to = "D"} : memref<64xf32>
      } {loop_name = "v", pipeline_ii = 1 : ui32}
    } {loop_name = "u", op_name = "outputs"}
    return
  }
}
