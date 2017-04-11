val cu = _import "test" public : int * MLton.Pointer.t -> unit;
val gen = _import "gen_host_send_ptr" public : unit -> MLton.Pointer.t;

val ptr = gen()
val _ = cu (5, ptr)
