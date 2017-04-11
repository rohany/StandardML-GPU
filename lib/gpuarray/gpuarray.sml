structure GPUArray = 
struct
  
  exception Range
  open Array
  
  (* cimport functions - these can be found in allocate.cu *)
  val gpu_alloc = _import "allocate_on_gpu" public : int * int -> MLton.Pointer.t;
  val gpu_free = _import "free_gpu_ptr" public : MLton.Pointer.t -> unit;
  val copy_int_from_gpu = 
    _import "copy_float_gpu" public : int array * MLton.Pointer.t * int -> unit;
  val copy_float_from_gpu = 
    _import "copy_float_gpu" public : real array * MLton.Pointer.t * int -> unit;


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

  fun toIntArray (a, size, ctype) = 
    let
      val hostarr = array(size, 0)
      val _ = copy_int_from_gpu(hostarr, a, size)
    in
      hostarr
    end

  fun toRealArray (a, size, ctype) = 
    let
      val hostarr = array(size, 0.0)
      val _ = copy_float_from_gpu(hostarr, a, size)
    in
      hostarr
    end

  (* requires that the sizes are the same *)
  fun copyIntoIntArray h (a, size, _) = 
    if length h <> size then raise Range else copy_int_from_gpu(h, a, size)

  (* requires that the sizes are the same *)
  fun copyIntoRealArray h (a, size, _) = 
    if length h <> size then raise Range else copy_float_from_gpu(h, a, size)

  fun getDevicePtr (a, _, _) = a

  fun getSize (_, a, _) = a
  
  fun getCtype (_, _, a) = a
   
end
