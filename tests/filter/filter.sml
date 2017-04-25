val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString(x)

structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence

val s1 = Seq.tabulate (fn i => i) size
val (sres, str1) = Timer.run (fn () => Seq.filter (fn i => i mod 2 = 0) s1)
val _ = print("SML : " ^ str1 ^ "\n")
val a1 = GPUSeq.tabulate GPUINTLambdas.identity size
val (res, str2) = Timer.run (fn () => GPUSeq.filter GPUINTLambdas.is_even a1)
val _ = print("SMLGPU : " ^ str2 ^ "\n")
val hostarr = GPUArray.toIntArray res
val hostseq = Seq.fromArray hostarr

val true = Seq.length hostseq = Seq.length sres

val vals = Seq.tabulate (fn i => Seq.nth hostseq i = Seq.nth sres i) (Seq.length sres)
val test = Seq.filter (fn x => x) vals
val _ = if Seq.length test = (Seq.length sres) then print("Success!\n") else print("Test Failed\n")
(*val _ = print(Printer.arrayToString Int.toString hostarr ^ "\n")*)
