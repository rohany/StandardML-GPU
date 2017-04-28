val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString x

structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence
structure Fused = FusedINTGPUSequence

val a1 = Seq.tabulate(fn i => 1) size

fun run () = 
  let
    val a = Seq.map (fn i => 2 * i) a1
    val a = Seq.map (fn i => 2 * i) a
    val a = Seq.map (fn i => 2 * i) a
    val a = Seq.map (fn i => 2 * i) a
    val a = Seq.map (fn i => 2 * i) a
    val a = Seq.map (fn i => 2 * i) a
    val a = Seq.map (fn i => 2 * i) a
  in
    a
  end
val (a1, str1) = Timer.run run
val _ = print("SML : " ^ str1 ^ "\n")




val a2 = GPUArray.initInt size 1

fun run () = 
  let
    val a = GPUSeq.map GPUINTLambdas.double a2
    val a = GPUSeq.map GPUINTLambdas.double a
    val a = GPUSeq.map GPUINTLambdas.double a
    val a = GPUSeq.map GPUINTLambdas.double a
    val a = GPUSeq.map GPUINTLambdas.double a
    val a = GPUSeq.map GPUINTLambdas.double a
    val a = GPUSeq.map GPUINTLambdas.double a
  in
    a
  end


val (res2, str2) = Timer.run run
val _ = print("SMLGPU : " ^ str2 ^ "\n")


fun run a () = 
  let
    val a = Fused.map GPUINTLambdas.double a
    val a = Fused.map GPUINTLambdas.double a
    val a = Fused.map GPUINTLambdas.double a
    val a = Fused.map GPUINTLambdas.double a
    val a = Fused.map GPUINTLambdas.double a
    val a = Fused.map GPUINTLambdas.double a
    val b = Fused.map GPUINTLambdas.double a
  in
    Fused.mapForce b
  end

val a4 = Fused.all 1 size

val (res2, str2) = Timer.run (run a4)

val hostarr = Fused.toArray res2

val _ = print("Fused Time : " ^ str2 ^ "\n")

(*val _ = print(Printer.arrayToString Int.toString hostarr ^ "\n")*)


(*val _ = if wei = size then print("Success!\n") else print("Test Failed\n")*)

