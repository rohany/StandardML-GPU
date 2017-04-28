structure FusedINTGPUSequence = 
struct
  
  structure GPUSeq = INTGPUSequence

  type 'a gpuseq = 'a GPUSeq.gpuseq * MLton.Pointer.t list

  val tabulate_cuda = 
    _import "tabulate_int" public : int * MLton.Pointer.t -> MLton.Pointer.t;
  val map_force_cuda = 
    _import "map_force" public : 
    MLton.Pointer.t * int * MLton.Pointer.t Array.array * int -> unit; 


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

  fun toArraySequence (s, l) = GPUSeq.toArraySequence s

  fun toArray (s, l) = GPUArray.toIntArray(s)

end
