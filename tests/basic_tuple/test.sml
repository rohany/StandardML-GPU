val [x] = CommandLine.arguments()
val SOME(size) = Int.fromString(x)

structure Seq = ArraySequence
structure GPUSeq = INTTUPLEGPUSequence

val s1 = Seq.tabulate (fn i => (2,2)) size
val (sres, str1) = Timer.run (fn () => Seq.nth s1 (size - 1))
val _ = print("SML : " ^ str1 ^ "\n")
val a1 = GPUSeq.tabulate (fn i => (2, 2)) size
val (res, str2) = Timer.run (fn () => GPUSeq.nth a1 (size - 1))
val _ = print("SMLGPU : " ^ str2 ^ "\n")
val hostarr = GPUArray.toIntArray res
val hostseq = Seq.fromArray hostarr

(*val vals = Seq.tabulate (fn i => Seq.nth hostseq i = Seq.nth sres i) size
val test = Seq.filter (fn x => x) vals*)

val _ = if (res = sres) then print("Success!\n") else print("Test Failed\n")
(* val _ = print(Printer.arrayToString Int.toString hostarr ^ "\n") *)
