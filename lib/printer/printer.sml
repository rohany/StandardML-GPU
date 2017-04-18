structure Printer = 
struct
  fun arrayToString f a = 
    let
      val L = Array.foldr (op ::) [] a
      fun listToString L = 
        let
          fun listToString' [] = ""
            | listToString' (x::[]) = f(x)
            | listToString' (x::L) = f(x) ^ ", " ^ listToString' L
        in
          "[" ^ listToString' L ^ "]"
        end
    in
      listToString L
    end
end
