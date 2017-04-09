## Project Proposal

# Summary
We are going to add high performance GPU bindings to the Standard ML language, to make it possible to run kernels over
device arrays from Standard ML. Additionally, we want to provide a high performace implementation of a subset of the
[SEQUENCE](http://www.cs.cmu.edu/~15210/docs/sig/sequence/SEQUENCE.html) library, so users can continue to write the high 
level algorithms they are used to while work is being sent to the GPU during runtime.

# Background
Functional programming very naturally describes parallel transformations of data, but doesn't yet have an efficient mechanisms
of mapping these declarative programs onto a parallel platform like a GPU. Since functional languages are very easy to write in, 
there should be some way for functional programmers to see real speedups in the code they write, and should be able to enjoy the 
computing power of the GPU without having to diverge too far from the purely functional, side-effect free mode of thinking. 

Functional algorithms can be expressed through the sequence interface above, where we can represent problems in terms of 
maps, scans, reduces etc. These algorithms all can be implemented efficiently on GPU's, and the programmer can use these
operations without worrying about how to distribute work onto the hardware. 

Our approach will involve using the foriegn function interface offered by the MLton compiler, to allow interfacing between
the SML, C and CUDA code. 

# Challenge
The challenge of this project is multifold : 
1. Actually getting Standard ML to interface correctly with CUDA and C
2. Being able to create an interface with CUDA that is not overly complicated, and still maintains a functional style
3. Implementing efficient mappings of declarative work to kernels
4. Maintaining invariants of functional programs like immutability from the outside, but allowing for high performance
   use of the hardware.

# Resources
We will be starting from scratch. We would like the course staff to install the MLton compiler on the Gates cluster machines - 
can be done with 'apt-get install mlton'.

# Goals and Deliverables
We plan to deliver an interface for running kernels on the GPU from a Standard ML interface, along with being able to provide
abstractions from memory transfer and management to the user. Additionally, we also plan to deliver a even higher level abstraction
of GPU primitives in the form of a thrust style sequence library where users can express thier algorithms in terms of high level
primitives, and our implementation will allow for fast execution. Lastly, we plan to optimize this implementation to allow 
fusing of operations to avoid the large amount of data copying that functional programs incur, and allow for users to pass around
arbitary higher order functions for use in thier sequence operations. 

If all goes well, we hope to implement some kind of polymorphism in this interface to allow for a more functional style. Additionally
we would like to implement scripts to automate some of the interface process, and a small transpiler for simple SML functions. 

At the parallelism competition, we would like to show comparisons of use between normal SML code, and SML code that can be run
on the GPU, and speedup comparisons between the two. 

# Platform Choice
Since functional programs naturally express a very data-parallel method of computation, a GPU is a natural platform to 
run the code on. Additionally, current research is going on in running this code on multi-core CPU's, so it would be interesting
to compare our implementation with that research. 

# Schedule

| Dates | Goal | Description |
| --- | --- | --- |
| April 10 | Setup | Learn about how to setup an interface between CUDA and SML, and begin implementation of kernel launch and GPU memory transfer infrastructure |
| April 17 | Sequences | Implement a fast serial sequence structure, and begin GPU implementations for basic types (int, float) |
| April 24 | Fast Sequences | Optimize the sequence operations as best as possible. The primitives' performance is key to the performance of the whole interface |
| May 1 | Optimizations | Add fusing of operations to sequence implementations |
| May 8 | Wrapping Up | Begin writeup and presentations, and if extra time work on the extra things detailed above |

