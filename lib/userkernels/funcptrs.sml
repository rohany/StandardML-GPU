structure GPUINTLambdas = 
struct
  local
    val gen_identity = _import "gen_identity" public : unit -> MLton.Pointer.t;
    val gen_double = _import "gen_double" public : unit -> MLton.Pointer.t;
  in
    val identity = gen_identity ()
    val double = gen_double ()
  end
end

structure GPUFLOATLambdas = 
struct
  val () = ()
end
