structure GPUKernels = 
struct
  open Real
  val mandel_gpu = 
    _import "mandel_gpu" public : MLton.Pointer.t * int * int * int * real *
    real * real * real -> unit;

end
