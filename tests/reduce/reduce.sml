val size = 100
structure Seq = ArraySequence
val a1 = Seq.tabulate(fn i => 1) size
val res1 = Seq.reduce (op +) 0 a1
val a2 = GPUArray.initInt size 1
val res2 = INTGPUSequenceNaive.reduce GPUINTLambdas.add 0 a2
val _ = print(Int.toString(res1) ^ " " ^ Int.toString(res2) ^ "\n")
val true = res1 = res2
