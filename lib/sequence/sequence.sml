structure ArraySequence = 
struct
  structure A = Array
  structure AS = ArraySlice
  
  type 'a seq = 'a AS.slice

  (* it is to be noted that these sequence functions
   * can additionally be implemented in parallel, but
   * we will be comparing against SML running on a single
   * core. For further experimentation, we can run against
   * parallel sequence implementations using the 
   * mlton-spoonhower compiler. *)

  fun empty () = AS.full(A.fromList [])
  
  fun tabulate f n = AS.full(A.tabulate(n, f))

  fun fromArray s = AS.full s

  fun nth s i = AS.sub(s, i)

  fun length s = AS.length s

  fun map f s = AS.full(A.tabulate(length s, fn i => f(nth s i)))

  fun reduce f b s = AS.foldr f b s

  fun filter p s = 
    AS.full (A.fromList (AS.foldr (fn (x, l) => if p x then x :: l else l) [] s))

  fun scan f b s = 
    let
      val a = AS.full(A.array(length s, b))
      fun loop i acc = 
        if i = length s then acc else
        let
          val x = nth s i
          val _ = AS.update(a, i, acc)
          val res = f(acc, x)
        in
          loop (i+1) res
        end
      val r = loop 0 b
    in
      (a, r)
    end
  
  fun scanIncl f b s = 
    let
      val a = AS.full(A.array(length s, b))
      fun loop i acc = 
        if i = length s then acc else
        let
          val x = nth s i
          val res = f(acc, x)
          val _ = AS.update(a, i, res)
        in
          loop (i+1) res
        end
      val r = loop 0 b
    in
      a
    end

  fun zipwith f (a, b) = AS.full(A.tabulate(length a, fn i => f(nth a i, nth b i)))

end
