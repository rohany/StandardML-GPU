structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence

structure SML_PARENS = 
struct

  fun gen_paren i = if i mod 2 = 0 then 1 else ~1
  
  fun match_paren_seq s1 () = 
    let
      val (s, last) = Seq.scan (op +) 0 s1
    in
      last = 0 andalso (Seq.reduce Int.min 0 s) >= 0
    end
  
  fun run_test size = 
    let
      val s1 = Seq.tabulate gen_paren size
      val (sres, str1) = Timer.run (match_paren_seq s1)
      val _ = print("SML : " ^ str1 ^ ", ") 
    in
      sres
    end

end

structure GPU_PARENS =
struct

  fun match_paren_gpu s2 () = 
    let
      val (s, last) = GPUSeq.scan GPUINTLambdas.add 0 s2
    in
      last = 0 andalso (GPUSeq.reduce GPUINTLambdas.min 0 s) >= 0
    end

  fun run_test size = 
    let
      val s2 = GPUSeq.tabulate GPUINTLambdas.parens size
      val (res, str2) = Timer.run (match_paren_gpu s2)
      val _ = print("SMLGPU : " ^ str2 ^ "\n")
    in
      res
    end

end


structure Main = 
struct
  
  fun run () = 
    let
      val x = List.hd (CommandLine.arguments())
      val size = Option.valOf(Int.fromString x)
      val (sml_res, gpu_res) = (SML_PARENS.run_test size, GPU_PARENS.run_test size)
      (*val _ = if sml_res = gpu_res then print("Test Passed\n") else print("Test Failed\n")*)
    in
      ()
    end
  (*)
  fun profile () = 
    let
      val x = ref 0
      val step = 100000000
    in
      while (!x < 2100000000) do (
        x := !x + step;
      let
        val (sml_res, gpu_res) = (SML_PARENS.run_test !x, GPU_PARENS.run_test !x)
        (*val _ = if sml_res = gpu_res then print("Test Passed\n") else print("Test Failed\n")*)
      in
        ()
      end
      )
    end
  *)
end

val () = Main.run ()
(* 
val () = Main.profile ()
*)
