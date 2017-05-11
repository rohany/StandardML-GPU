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
