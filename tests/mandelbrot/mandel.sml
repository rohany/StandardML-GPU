structure CPUMandel = 
struct
  
  fun mandel c_re c_im count = 
    let
      fun mandel' i z_re z_im = 
        if z_re * z_re + z_im * z_im > 4.0 then i else
        if i = count then i else
        let
          val new_re = z_re * z_re - z_im * z_im
          val new_im = 2.0 * z_re * z_im
          val z_re' = c_re + new_re
          val z_im' = c_im + new_im
        in
          mandel' (i+1) z_re' z_im'
        end
    in
      mandel' 0 c_re c_im
    end

  fun runMandel () = 
    let
      val (x0, x1, y0, y1) = (~2.0, 1.0, ~1.0, 1.0)
      val (width, height) = (1200.0, 800.0)
      val (dx, dy) = ((x1 - x0) / width, (y1 - y0) / height)
      val maxIter = 256
      fun loop i = 
        let
          val x = x0 + Real.fromInt(i mod Real.floor width) * dx
          val y = y0 + Real.fromInt(i div Real.floor width) * dy
        in
          mandel x y maxIter
        end
    in
      Array.tabulate(Real.floor width * Real.floor height, loop)
    end

end

structure GPUMandel = 
struct
  fun runMandel () = 
    let
      val (x0, x1, y0, y1) = (~2.0, 1.0, ~1.0, 1.0)
      val (width, height) = (1200.0, 800.0)
      val (dx, dy) = ((x1 - x0) / width, (y1 - y0) / height)
      val maxIter = 256
      val gpuarr = 
        GPUArray.init (Real.floor width * Real.floor height) (CTYPES.CINT)
      val _ = GPUKernels.mandel_gpu
              (GPUArray.getDevicePtr gpuarr, maxIter, Real.floor width, 
               Real.floor height, dx, dy, x0, y0)
    in
      GPUArray.toIntArray gpuarr
    end

end

structure Main = 
struct
  fun run () = 
    let
      val (res1, time) = Timer.run CPUMandel.runMandel
      val _ = print("SML time " ^ time ^ "\n")
      val (res2, time) = Timer.run GPUMandel.runMandel
      val _ = print("SMLGPU time " ^ time ^ "\n")
      val bools = List.tabulate
          (Array.length res1, fn i => if Array.sub(res1, i) = Array.sub(res2,i)
                                      then true else
                                        ((*print(Int.toString(Array.sub(res1, i))
                                        ^ " " ^ Int.toString(Array.sub(res2, i))
                                        ^ " " ^ Int.toString(i) ^"\n");*)false))
      val true = List.all (fn x => x) (bools)
    in
      ()
    end
end

val _ = Main.run()
