structure GPUINTTUPLELambdas = 
struct
  
  val gen_const1_int_tuple = 
    _import "gen_const1_int_tuple" public : unit -> MLton.Pointer.t;


  val const1 = gen_const1_int_tuple ()  

end
