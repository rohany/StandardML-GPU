structure FusedINTGPUSequence = 
struct
  
  structure GPUSeq = INTGPUSequence

  type 'a gpuseq = 'a GPUSeq.gpuseq * MLton.Pointer.t list

  val tabulate_cuda = 
    _import "tabulate_int" public : int * MLton.Pointer.t -> MLton.Pointer.t;
  val map_force_cuda = 
    _import "map_force" public : 
    MLton.Pointer.t * int * MLton.Pointer.t Array.array * int -> unit;
  val reduce_cuda = 
    _import "fused_reduce_int_shfl" public : 
    MLton.Pointer.t * int * int * MLton.Pointer.t * MLton.Pointer.t Array.array
    * int -> int;
  val incl_scan_cuda = 
    _import "fused_inclusive_scan_int" public : 
    MLton.Pointer.t * MLton.Pointer.t * int * int * MLton.Pointer.t Array.array
    * int -> MLton.Pointer.t;
  val excl_scan_cuda = 
    _import "fused_exclusive_scan_int" public : 
    MLton.Pointer.t * MLton.Pointer.t * int * int * MLton.Pointer.t Array.array
    * int -> int;
  val filter_cuda = 
    _import "fused_filter_int" public : 
    MLton.Pointer.t * int * MLton.Pointer.t * int ref * MLton.Pointer.t
    Array.array  * int -> MLton.Pointer.t;
  val zipwith_cuda = 
    _import "fused_zipwith_int" public : 
    MLton.Pointer.t * MLton.Pointer.t * MLton.Pointer.t * int * MLton.Pointer.t
    Array.array * int * MLton.Pointer.t Array.array * int -> MLton.Pointer.t;

  
  (* it is possible for us to add stuff like fusing more than just
   * map operations, but for starters, we will fuse maps *)

  fun listToArr l = Array.fromList(List.rev l)

  val null = MLton.Pointer.null;

  fun all b n = (GPUSeq.all b n, [])

  fun tabulate f n = 
    (GPUSeq.tabulate f n , [])

  (* this is the fuse operator *)
  fun map f (s, l) = (s, f::l)

  fun mapForce (s as (a,n,_), l) = 
    let
      val arr = listToArr l
      val () = map_force_cuda(a, n, arr, Array.length arr)
    in
      (s, [])
    end

  (* reduce will force the computation as well 
   * as perform the reduction *)
  fun reduce f b ((a,n,_), l) =
    let
      val funcs = listToArr l
    in
      reduce_cuda(a, n, b, f, funcs, Array.length funcs)
    end

  fun scan f b ((a, n, _), l) = 
    let
      val funcs = listToArr l
      val res = incl_scan_cuda(a, f, n, b, funcs, Array.length funcs)
    in
      (((a, n, CTYPES.CINT), []), res)
    end


  fun scanIncl f b ((a, n, _), l) = 
    let
      val funcs = listToArr l
      val res = incl_scan_cuda(a, f, n, b, funcs, Array.length funcs)
    in
      ((res, n, CTYPES.CINT), [])
    end

  fun filter p ((a, n, _), l) = 
    let
      val outlen = ref 0
      val funcs = listToArr l
      val a' = filter_cuda(a, n, p, outlen, funcs, Array.length funcs)
    in
      ((a', !outlen, CTYPES.CINT), [])
    end
  
  fun zipwith f ((a1, n, _), l1) ((a2, _, _), l2) = 
    let
      val funcs1 = listToArr l1
      val funcs2 = listToArr l2
      val out = zipwith_cuda(a1, a2, f, n, funcs1, Array.length funcs1, funcs2,
      Array.length funcs2)
    in
      ((out, n, CTYPES.CINT), [])
    end

  fun toArraySequence (s, l) = GPUSeq.toArraySequence s

  fun toArray (s, l) = GPUArray.toIntArray(s)

end
