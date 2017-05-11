structure Seq = ArraySequence
structure GPUSeq = INTGPUSequence
structure L = GPUINTLambdas

val int_max = Option.valOf(Int.maxInt)
val int_min = Option.valOf(Int.minInt)

structure CPU_MCSS = 
struct

  fun mcss S = 
    let
      val sums = Seq.scanIncl (op +) 0 S
      val (mins, _) = Seq.scan Int.min int_max sums
      val gains = Seq.zipwith (op -) (sums, mins)
    in
      Seq.reduce Int.max int_min gains
    end

end

structure GPU_MCSS = 
struct

  fun mcss S = 
    let
      val sums = GPUSeq.scanIncl L.add 0 S
      val m_sums = GPUSeq.makeCopy sums
      val (mins, _) = GPUSeq.scan L.min int_max m_sums
      val gains = GPUSeq.zipwith L.sub (sums, mins)
    in
      GPUSeq.reduce L.max int_min gains
    end

end


structure Main = 
struct
  
  fun r i =
    let
      val w = Word.mod(MLton.Random.rand(),Word.fromInt(5))
    in
      Word.toInt(w)
    end

  fun gen_rand_array n = 
    Array.tabulate(n, r)

  val size = Option.valOf(Int.fromString(List.hd(CommandLine.arguments())))
  
  val toRun = gen_rand_array size
  val s = Seq.tabulate (fn i => Array.sub(toRun, i)) size
  val gs = GPUSeq.fromArray toRun

  fun run () = 
    let
      val (r1, str) = Timer.run (fn () => CPU_MCSS.mcss s)
      val _ = print("SML : " ^ str ^ "\n")
      val (r2, str) = Timer.run (fn () => GPU_MCSS.mcss gs)
      val _ = print("SML : " ^ str ^ "\n")
      val _ = if r1 = r2 then print("Success\n") else print("Failed\n")
    in
      ()
    end 

end

val _ = Main.run()
