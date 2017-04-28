structure GPUINTLambdas = 
struct
  val gen_identity = _import "gen_identity_int" public : unit -> MLton.Pointer.t;
  val gen_double = _import "gen_double_int" public : unit -> MLton.Pointer.t;
  val gen_add = _import "gen_add_int" public : unit -> MLton.Pointer.t;
  val gen_is_even = _import "gen_is_even_int" : unit -> MLton.Pointer.t;
  val gen_left = _import "gen_left_int" public : unit -> MLton.Pointer.t;
  
  val identity = gen_identity ()
  val double = gen_double ()
  val add = gen_add () 
  val is_even = gen_is_even ()
  val left = gen_left()
  
end

structure GPUFLOATLambdas = 
struct
  open Real
  val () = ()
end
