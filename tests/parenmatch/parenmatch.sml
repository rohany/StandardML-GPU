val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString(x)

structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence

fun gen_paren i = if i mod 2 = 0 then 1 else ~1

val s1 = Seq.tabulate gen_paren size

fun match_paren_seq () = 
  let
    val (s, last) = Seq.scan (op +) 0 s1
  in
    last = 0 andalso (Seq.reduce Int.max 0 s) >= 0
  end

val (sres, str1) = Timer.run match_paren_seq
val _ = print("SML : " ^ str1 ^ "\n")

val s2 = GPUSeq.tabulate GPUINTLambdas.parens size

fun match_paren_gpu () = 
  let
    val (s, last) = GPUSeq.scan GPUINTLambdas.add 0 s2
  in
    last = 0 andalso (GPUSeq.reduce GPUINTLambdas.max 0 s) >= 0
  end


val (res, str2) = Timer.run match_paren_gpu
val _ = print("SMLGPU : " ^ str2 ^ "\n")

val _ = if res = sres then print("Success!\n") else print("Test Failed\n")

