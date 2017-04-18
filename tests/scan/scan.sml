val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString(x)

structure Seq = ArraySequence
structure GPUSeq = INTGPUSequenceNaive

val s1 = Seq.tabulate (fn i => 2) size
val (sres, str1) = Timer.run (fn () => Seq.scanIncl (op +) 0 s1)
val _ = print("SML : " ^ str1 ^ "\n")
val a1 = GPUSeq.all 2 size
val (res, str2) = Timer.run (fn () => GPUSeq.scanIncl GPUINTLambdas.add 0 a1)
val _ = print("SMLGPU : " ^ str2 ^ "\n")
val hostarr = GPUArray.toIntArray res
val hostseq = Seq.fromArray hostarr

val vals = Seq.tabulate (fn i => Seq.nth hostseq i = Seq.nth sres i) size
val test = Seq.filter (fn x => x) vals
val _ = if Seq.length test = size then print("Success!\n") else print("Test Failed\n")
(* val _ = print(Printer.arrayToString Int.toString hostarr ^ "\n") *)
