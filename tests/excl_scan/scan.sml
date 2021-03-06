val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString(x)

structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence

val s1 = Seq.tabulate (fn i => 1) size
val ((sres, sv), str1) = Timer.run (fn () => Seq.scan (op +) 0 s1)
val _ = print("SML : " ^ str1 ^ "\n")
val a1 = GPUSeq.all 1 size
val ((res, gv), str2) = Timer.run (fn () => GPUSeq.scan GPUINTLambdas.add 0 a1)
val _ = print("SMLGPU : " ^ str2 ^ "\n")
val hostarr = GPUArray.toIntArray res
val hostseq = Seq.fromArray hostarr

val vals = Seq.tabulate (fn i => if Seq.nth hostseq i = Seq.nth sres i then true
                                 else (print(Int.toString i ^ "\n");false)) size
val test = Seq.filter (fn x => x) vals
val _ = if Seq.length test = size andalso gv = sv 
        then print("Success!\n") else print("Test Failed\n")
(*val _ = print(Int.toString(sv) ^ " " ^ Int.toString(gv) ^ "\n")
val _ = print(Printer.arrayToString Int.toString hostarr ^ "\n")*)
