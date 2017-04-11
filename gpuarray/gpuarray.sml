structure GPUArray = 
struct
  
  exception NYI
  
  val gpu_alloc = _import "allocate_on_gpu" public : int * int -> MLton.Pointer.t;
  val gpu_free = _import "free_gpu_ptr" public : MLton.Pointer.t -> unit;

  (* we hold the pointer to the array, the size, and the type *)
  type 'a gpuarray = MLton.Pointer.t * int * int
  
  (* initializes an array on the GPU and sets it to zero *)
  fun init size ctype = 
    let
      val gpuptr = gpu_alloc (size, ctype)
    in
      (gpuptr, size, ctype)
    end

  fun destroy (a, _, _) = gpu_free a 

  (* this function is probably going to have to incur double
   * the memory traffic due to the constraints of the type system *)
  fun toArray (a, size, ctype) = raise NYI

  fun getDevicePtr (a, _, _) = a

  fun getSize (_, a, _) = a
  
  fun getCtype (_, _, a) = a
   
end
