val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString x
structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence
val a1 = Seq.tabulate(fn i => i) size
val (a1, str1) = Timer.run (fn () => Seq.map (fn i => 2) a1)
val _ = print("SML : " ^ str1 ^ "\n")
val a2 = GPUArray.initInt size 1
val (res2, str2) = Timer.run (fn () => GPUSeq.map GPUINTLambdas.double a2)
val _ = print("SMLGPU : " ^ str2 ^ "\n")
val a3 = GPUSeq.toArraySequence res2
val wei = Seq.length (Seq.filter (fn x => x = 2) a3)
val _ = if wei = size then print("Success!\n") else print("Test Failed\n")

