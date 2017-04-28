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

  
  (* it is possible for us to add stuff like fusing more than just
   * map operations, but for starters, we will fuse maps *)

  fun listToArr l = Array.fromList(List.rev l)


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

  fun toArraySequence (s, l) = GPUSeq.toArraySequence s

  fun toArray (s, l) = GPUArray.toIntArray(s)

end
