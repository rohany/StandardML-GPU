structure INTGPUSequenceNaive = 
struct
  
  open GPUArray

  type 'a gpuseq = 'a gpuarray 
  
  val tabulate_cuda = 
    _import "tabulate_int" public : int * MLton.Pointer.t -> MLton.Pointer.t;
  val map_cuda = 
    _import "map_int" public : MLton.Pointer.t * MLton.Pointer.t * int -> MLton.Pointer.t;
  val reduce_cuda = 
    _import "reduce_int_shfl" public : MLton.Pointer.t * int * int * MLton.Pointer.t -> int;

  fun all b n = initInt n b

  fun tabulate f n ctype = 
    let
      val a = tabulate_cuda (n, f)
    in
      (a, n, CTYPES.CINT)
    end
  
  (* this is a destructive mapping operation *)
  fun map f (a, n, _) = 
    let
      val a' = map_cuda(a, f, n)
    in
      (a', n, CTYPES.CINT)
    end

  val makeCopy = copy 

  val length = getSize

  fun toArraySequence s = ArraySlice.full(toIntArray s)

  fun reduce f b (a, n, _) = reduce_cuda(a, n, b, f)

  val scan = ()

  val filter = ()

  val zip = ()

end
