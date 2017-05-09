## Final Writeup

# Overview
We developed StandardML-GPU, a Standard ML library and extension that allows a user to 
interface with CUDA, and allow Standard ML to take advantage of the computing power of the GPU. 
Our library provides an interface between raw CUDA code and Standard ML, an abstraction from
C/CUDA memory management and transfer, and a series of `SEQUENCE` data structre
implementations, which will be discussed in the report. Using our library, a user
is able to express thier algorithm in terms of high level operations like `scan`, `tabulate`, 
and `reduce`, and is able beat out **all other options** aside from handwritten CUDA. 

# Background
Functional programming naturally describes transformations of data in a very declarative manner.
Since functional languages are extremely easy for programmers to express algorithmic ideas in, 
they should also be able to see thier code run fast and efficiently without having to translate 
thier code into another language. Additionally, they should be able to use the same functional 
programming methodology and see good performance without drastic changes to thier own code in 
Standard ML. 

Functional programs expressed with [`SEQUENCE`](http://www.cs.cmu.edu/~15210/docs/sig/sequence/SEQUENCE.html) 
primitives allow for powerful abstractions for algorithm design, and leaves the dirty work of 
efficiently implementing these primitives up to the implementer of the library. Primitives like
`scan`, `reduce`, `filter`, etc. are extremely data parallel, and map well to the GPU platform. 
However, there previously was no way that functional programmers could use these ideas from a functional
setting, having to resort to using libraries like [Thrust](http://docs.nvidia.com/cuda/thrust/#axzz4gcJAv4tP) in 
C to get the same kind of abstraction. 

Allowing for an interface between Standard ML and the GPU has a number of difficulties, which 
relate to the restricted nature of GPU computation, in contrast to the lack of restrictions in 
terms of memory management and syntactic constructs of Standard ML. To be specific : 
1. Providing an intuitive abstraction for memory management for device memory, since Standard ML does not have manual memory management.
2. 
3.
4.


## Checkpoint
Find our checkpoint write-up [here](checkpoint.md).

## Proposal
Find our proposal write-up [here](proposal.md).
