structure Timer = 
struct
  
  fun run f = 
    let
      val startTime = Time.now()
      val result = f()
      val endTime = Time.now()
      val elapsed = Time.- (endTime, startTime)
    in
      (result, "Running Time : " ^ Time.fmt 4 elapsed)
    end

end
