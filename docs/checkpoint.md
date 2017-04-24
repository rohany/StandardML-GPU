## Checkpoint Report

# Progress Review
So far, we have completed a large amount of the structure and framework for the interface. We have completed
* A GPUArray structure and backend that allows a SML user to create and manipulate arrays on the GPU, and allows 
  interfacing with the SML builtin Array structure, so a user can copy to and from SML arrays. Additionally, this
  structure allows a user to write their own CUDA kernels, and then call these kernels on the GPUArrays. 
* A fast serial Sequence implementation to benchmark against. 
* Testing frameworks to assure the interface between SML and CUDA (memory copying, kernel launches) are functioning properly. 
* Begun implementation of Sequence primitives for integers in CUDA. So far we have implemented tabulate, map, reduce, and scan. 

A large amount of time went into understanding the quirks of the SML foreign function interface, and trying to design 
an interface around this that was easily usable, preserved a functional style, and would limit how much a user would 
need to leave SML and write in CUDA/C. Additionally, a large amount of time was spent trying to get the compiled code 
from the two different languages to link together, and correctly transfer control between the languages. 

# Preliminary Results
Our work so far allows for two different styles of using this interface. The first is to simply just write 
CUDA kernels, and run them on arrays using the GPUArray interface, similar to the way that PyCUDA works. 
An example of this method can be found in tests/mandelbrot/mandel.sml, and the kernel in lib/userkernels/mandel.cu. 
(include a code sample of this later). This method is mainly for operations not included in our library, or
for writing more specific kernels than given by our library. 

The second way is to use the structures in lib/gpusequence. Here, we define a Sequence data structure with a 
set of operations designed to run efficiently in parallel on the GPU. These structures are similar
to the Thrust interface, but due to restrictions of the SML foreign function interface, cannot be polymorphic. Here 
is an example of how the current interface looks, in comparison to an SML implementation of Sequences. 

~~~~ocaml
structure Seq = ArraySequence
structure GPUSeq = INTGPUSequenceNaive
structure Lambdas = GPUINTLambdas
(* SML SEQUENCE EXAMPLE *)

val init = Seq.tabulate (fn i => i) 100000000
val scanned = Seq.scanIncl (op +) 0 init

(* SMLGPU SEQUENCE EXAMPLE *)

val init = GPUSeq.tabulate Lambdas.identity 100000000
val scanned = GPUSeq.scanIncl Lambdas.add 0 init

~~~~
The semantics of the two versions are nearly identical, aside from being able to inline functions in the sequence version - 
due to restrictions of CUDA function pointers, functions to be run on the GPU need to be pre-compiled separately and 
must go through a small process of copying GPU addresses to the host, which is why the functions are written separately 
and imported in the GPULambdas structures.

We have run preliminary tests of our GPU implementations of primitives against our fast serial implementation, thrust, and 
current research work that allows SML to run in parallel on multicore CPU's. 

For an input of size 100000000, we find achieved the following results for our sequence implementation thus far: 

| Implementation | Scan Time | Reduce Time |
| --- | --- | --- |
| Vanilla SML | 0.24 seconds| 0.1041 seconds |
| Thrust | 0.0120 seconds | 0.0060 seconds |
| SML-GPU | 0.0129 seconds | 0.0076 seconds |

These results are very promising - even with the overhead of using device function pointers, and having to jump from SML 
to CUDA during runtime, we are basically running nearly as fast as Thrust, while writing what looks like normal SML. 
The major price to pay is that we lose some functional features like polymorphism and variable bindings. Furthermore, 
we are able to perform faster than thrust for specific reductions such as sum, however we have found the flexibility 
of function pointers is worth the tradeoff.

# Goals and Deliverables
We think we are on track to deliver a CUDA interface for SML and a fast sequence library.  We believe we will be 
able to finish this, and can start implementing operation fusing, and support for more sequence types, such as
integer pairs or triples. 

We will show at the parallelism competition code demos and performance graphs on a variety of problems, comparing against
single core and multi core SML, as well as thrust. We hope to be able to say that we are able to offer thrust-level or
faster performance in SML, and that we beat multicore parallelism on large core counts as well. We also hope that this is 
acheveable with a nearly drop-in-place interface.

# Revised Schedule

| Dates | Goal | Description |
| --- | --- | --- |
| April 10 | Setup | Completed : Learn about how to setup an interface between CUDA and SML, and begin implementation of kernel launch and GPU memory transfer infrastructure |
| April 17 | Sequences | Completed : Implement a fast serial sequence structure, and begin GPU implementations for basic types (int, float) |
| April 24 | Fast Sequences | We are ahead of schedule here, and instead of starting with basic GPU implementations, we went ahead and decided to implement optimized versions of the primitives upfront. What is left is to finish implementations of filter and zip, and write the library implementation for floating point values. We can add more primitives later. This task is for Rohan and Brandon. |
| May 1 | Optimizations | Add fusing of operations to sequence implementations, and add support for tuples. Both tasks for Rohan and Brandon. |
| May 8 | Wrapping Up | Begin writeup and presentations, and write performance benchmarking tests. For Rohan and Brandon |

# Issues
We are running into an issue with compiling and linking device code across files - no matter what, we either run into issues 
in runtime, or are unable to link the device compiled code with the compiled SML. If we cannot figure out how to get
CUDA to compile and link in this way, we are unsure how we can implement the auto sequence operation fusing we were planning
to implement in our proposal. 

Additionally, we were unable to figure out a good way to implement any sort of polymorphism in our library, due to the restrictions
of typing in C (cannot use templates, and void\* is not the answer to everything), 
and the foreign function interface of SML. To compensate, we have to make a few copies of our library
implementations to support different types, and figure out the best way to support tuples.
