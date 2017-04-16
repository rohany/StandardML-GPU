val gpuarr = GPUArray.initInt 10 5
val hostarr = GPUArray.toIntArray gpuarr
val _ = print(Int.toString(Array.sub(hostarr, 0)) ^ "\n")
