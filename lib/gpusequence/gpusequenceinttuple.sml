structure INTGPUSequence = 
struct
  
  open GPUArray

  type 'a gpuseq = 'a gpuarray 
  
  val tabulate_cuda = 
    _import "tabulate_intxint" public : int * MLton.Pointer.t -> MLton.Pointer.t;
  val map_cuda = 
    _import "map_intxint" public : MLton.Pointer.t * MLton.Pointer.t * int -> MLton.Pointer.t;
  val reduce_cuda = 
    _import "reduce_intxint_shfl" public : MLton.Pointer.t * int * int * MLton.Pointer.t -> int;
  val incl_scan_cuda = 
    _import "inclusive_scan_intxint" public : 
    MLton.Pointer.t * MLton.Pointer.t * int * int -> MLton.Pointer.t;
  val excl_scan_cuda = 
    _import "exclusive_scan_intxint" public : 
    MLton.Pointer.t * MLton.Pointer.t * int * int -> int;
  val filter_cuda = 
    _import "filter_intxint" public : 
    MLton.Pointer.t * int * MLton.Pointer.t * int ref -> MLton.Pointer.t;
  val zipwith_cuda = 
    _import "zipwith_intxint" public : 
    MLton.Pointer.t * MLton.Pointer.t * MLton.Pointer.t * int -> MLton.Pointer.t;

  fun all b n = initInt n b

  fun tabulate f n = 
    let
      val a = tabulate_cuda (n, f)
    in
      (a, n, CTYPES.CINTxINT)
    end
  
  (* this is a destructive mapping operation *)
  fun map f (a, n, _) = 
    let
      val a' = map_cuda(a, f, n)
    in
      (a', n, CTYPES.CINTxINT)
    end

  val makeCopy = copy 

  val length = getSize

  fun toArraySequence s = ArraySlice.full(toIntArray s)

  fun reduce f b (a, n, _) = reduce_cuda(a, n, b, f)

  fun scan f b (a, n, _) = 
    let
      val res = excl_scan_cuda(a, f, n, b)
    in
      ((a, n, CTYPES.CINTxINT), res)
    end

  (* unsure what happens when the compiler thinks 
   * this MLton.Pointer.t goes out of scope *)
  fun scanIncl f b (a, n, _) = 
    let
      val a' = incl_scan_cuda(a, f, n, b)
    in
      (a', n, CTYPES.CINTxINT)
    end
  
  fun filter p (a, n, _) = 
    let
      val outlen = ref 0
      val a' = filter_cuda(a, n, p, outlen)
    in
      (a', !outlen, CTYPES.CINTxINT)
    end
  
  (* requires both have the same length *)
  fun zipwith f (a1, n, _) (a2, _, _) =
    let
      val out = zipwith_cuda(a1, a2, f, n)
    in
      (out, n, CTYPES.CINTxINT)
    end

end
