structure GPUArray = 
struct
  
  exception Range
  open Array
  open Real
  
  (* cimport functions - these can be found in allocate.cu *)
  val init_gpu = _import "init_gpu" public : unit -> unit;
  val gpu_alloc = _import "allocate_on_gpu" public : int * int -> MLton.Pointer.t;
  val gpu_free = _import "free_gpu_ptr" public : MLton.Pointer.t -> unit;
  val copy_int_from_gpu = 
    _import "copy_int_gpu" public : int array * MLton.Pointer.t * int -> unit;
  val copy_float_from_gpu = 
    _import "copy_float_gpu" public : real array * MLton.Pointer.t * int -> unit;
  val copy_int_into_gpu = 
    _import "copy_int_into_gpu" public : int array * int -> MLton.Pointer.t;
  val copy_float_into_gpu = 
    _import "copy_float_into_gpu" public : real array * int -> MLton.Pointer.t;
  val initwith_int = 
    _import "initInt_gpu" public : int * int -> MLton.Pointer.t;
  val initwith_float = 
    _import "initFloat_gpu" public : int * real -> MLton.Pointer.t;
  val copy_gpu = 
    _import "copy" public : MLton.Pointer.t * int * int -> MLton.Pointer.t;


  (* we hold the pointer to the array, the size, and the type *)
  type 'a gpuarray = MLton.Pointer.t * int * int

  fun INITCUDA () = init_gpu()
  
  (* initializes an array on the GPU and sets it to zero *)
  fun init size ctype = 
    let
      val gpuptr = gpu_alloc (size, ctype)
    in
      (gpuptr, size, ctype)
    end

  (* TODO add a init with argument to init as - can do in sequence as well *)

  fun initInt size b = 
    let
      val ptr = initwith_int(size, b)
    in
      (ptr, size, CTYPES.CINT)
    end
  
  fun initFloat size b = 
    let
      val ptr = initwith_float(size, b)
    in
      (ptr, size, CTYPES.CFLOAT)
    end

  fun destroy (a, _, _) = gpu_free a 

  fun copy (a, n, c) = (copy_gpu(a,n,c), n, c)

  (* is there a bug here? *)
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
  
  (* add copy into gpu *)
  fun fromIntArray a = 
    let
      val gpuarr = copy_int_into_gpu(a, length a)
    in
      (gpuarr, length a, CTYPES.CINT)
    end
  
  (* add copy into gpu *)
  fun fromRealArray a = 
    let
      val gpuarr = copy_float_into_gpu(a, length a)
    in
      (gpuarr, length a, CTYPES.CFLOAT)
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

val _ = GPUArray.INITCUDA()
