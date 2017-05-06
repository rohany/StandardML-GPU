val b = Array.array(1, 1)
val a = GPUArray.initInt 1 0
val _ = GPUArray.copyIntoIntArray b a 
val _ = print(Int.toString(Array.sub(b, 0)) ^ "\n")
val _ = GPUArray.destroy(a)
val _ = print("Hello!\n")
