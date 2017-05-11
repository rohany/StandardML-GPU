structure GPUINTLambdas = 
struct

  val gen_identity = _import "gen_identity_int" public : unit -> MLton.Pointer.t;
  val gen_double = _import "gen_double_int" public : unit -> MLton.Pointer.t;
  val gen_add = _import "gen_add_int" public : unit -> MLton.Pointer.t;
  val gen_is_even = _import "gen_is_even_int" : unit -> MLton.Pointer.t;
  val gen_left = _import "gen_left_int" public : unit -> MLton.Pointer.t;
  val gen_parens = _import "gen_paren_gen" public : unit -> MLton.Pointer.t;
  val gen_max = _import "gen_max_int" public : unit -> MLton.Pointer.t;
  val gen_min = _import "gen_min_int" public : unit -> MLton.Pointer.t;
  
  val identity = gen_identity ()
  val double = gen_double ()
  val add = gen_add () 
  val is_even = gen_is_even ()
  val left = gen_left()
  val parens = gen_parens()
  val max = gen_max()
  val min = gen_min()


	val gen_multiply = _import "gen_multiply_int" public : unit -> MLton.Pointer.t;
	val multiply = gen_multiply()


	val gen_sub = _import "gen_sub_int" public : unit -> MLton.Pointer.t;
	val sub = gen_sub()

end
