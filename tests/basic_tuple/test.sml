val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString(x)

structure Seq = ArraySequence
structure GPUSeq = INTTUPLEGPUSequence

val s1 = fn () => Seq.tabulate (fn i => (1, 1)) size
val (sres, str1) = Timer.run s1
val _ = print("SML : " ^ str1 ^ "\n")
val a1  = fn () => GPUSeq.tabulate GPUINTTUPLELambdas.const1 size
val (res, str2) = Timer.run a1
val _ = print("SMLGPU : " ^ str2 ^ "\n")
val hostarr = GPUSeq.toArray res
val hostseq = Seq.fromArray hostarr

val vals = Seq.tabulate (fn i => Seq.nth hostseq i = Seq.nth sres i) size
val test = Seq.filter (fn x => x) vals

val _ = if (Seq.length vals = size) then print("Success!\n") else print("Test Failed\n")
(* val _ = print(Printer.arrayToString Int.toString hostarr ^ "\n") *)
