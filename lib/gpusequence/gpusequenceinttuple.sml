structure INTTUPLEGPUSequence = 
struct
  
  open GPUArray

  type 'a gpuseq = MLton.Pointer.t * MLton.Pointer.t * int
  
  val tabulate_cuda = 
    _import "tabulate_int_tuple" public : 
    int * MLton.Pointer.t * MLton.Pointer.t ref * MLton.Pointer.t ref -> unit;
  val map_cuda = 
    _import "map_int_tuple" public : 
    MLton.Pointer.t * MLton.Pointer.t * MLton.Pointer.t * int -> unit;
  val reduce_cuda = 
    _import "reduce_int_tuple_shfl" public : 
    MLton.Pointer.t * MLton.Pointer.t * int *int * int * MLton.Pointer.t  * int
    ref * int ref -> unit;
  val incl_scan_cuda = 
    _import "inclusive_scan_intxint" public : 
    MLton.Pointer.t * MLton.Pointer.t * MLton.Pointer.t * int * int * int -> unit;
  val excl_scan_cuda = 
    _import "exclusive_scan_int_tuple" public : 
    MLton.Pointer.t * MLton.Pointer.t * MLton.Pointer.t * int * int * int * int ref *
    int ref -> unit;
  (*val filter_cuda = 
    _import "filter_intxint" public : 
    MLton.Pointer.t * MLton.Pointer.t * int * MLton.Pointer.t * 
    MLton.Pointer.t ref * MLton.Pointer.t ref * int ref -> unit;
  val zipwith_cuda = 
    _import "zipwith_intxint" public : 
    MLton.Pointer.t * MLton.Pointer.t * MLton.Pointer.t * int ->
    MLton.Pointer.t;*)

  (* fun all b n = initInt n b *)

  fun tabulate f n = 
    let
      val a1 = ref (MLton.Pointer.null)
      val a2 = ref (MLton.Pointer.null)
      val () = tabulate_cuda (n, f, a1, a2)
    in
      (!a1, !a2, n)
    end
  
  (* this is a destructive mapping operation *)
  fun map f (s as (a1, a2, n)) = 
    let
      val () = map_cuda(a1, a2, n, f)
    in
      s
    end
  
  (* TODO
  val makeCopy = copy 

  val length = getSize
  *)

  (*fun toArraySequence s = ArraySlice.full(toIntArray s)*)

  fun reduce f (b1, b2) (a1, a2, n) =
    let
      val o1 = ref 0
      val o2 = ref 0
      val _ = reduce_cuda(a1, a2, n, b1, b1, f, o1, o2)
    in
      (!o1, !o2)
    end

  fun scan f (b1, b2) (a1, a2, n) = 
    let
      val o1 = ref 0
      val o2 = ref 0
      val () = excl_scan_cuda(a1, a2, f, n, b1, b2, o1, o2)
    in
      ((a1,a2,n), (!o1, !o2))
    end

  (* unsure what happens when the compiler thinks 
   * this MLton.Pointer.t goes out of scope *)
  fun scanIncl f (b1, b2) (a1, a2, n) = 
    let
      val () = incl_scan_cuda(a1, a2, f, n, b1, b2)
    in
      (a1, a2, n)
    end
  
  fun toArray (a1, a2, n) = 
    let
      val ga1 = (a1, n, CTYPES.CINT)
      val ga2 = (a2, n, CTYPES.CINT)
      val ha1 = GPUArray.toIntArray ga1
      val ha2 = GPUArray.toIntArray ga2
      val out = Array.tabulate(n, fn i => (Array.sub(ha1, i), Array.sub(ha2, i)))
    in
      out
    end

  fun destroy (a1, a2, n) = (GPUArray.gpu_free(a1);GPUArray.gpu_free(a2))
  (*)
  fun filter p (a1, a2, n) = 
    let
      val o1 = ref(MLton.Pointer.null)
      val o2 = ref(MLton.Pointer.null)
      val outlen = ref 0
      val () = filter_cuda(a1, a2, n, p, o1, o2, outlen)
    in
      (!o1, !o2, !outlen)
    end*)
  (*)
  (* requires both have the same length *)
  fun zipwith f (a1, n, _) (a2, _, _) =
    let
      val out = zipwith_cuda(a1, a2, f, n)
    in
      (out, n, CTYPES.CINTxINT)
    end*)

end
