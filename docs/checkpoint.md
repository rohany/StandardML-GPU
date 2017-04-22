## Checkpoint Report

# Progress Review
So far, we have completed a large amount of the structure and framework for the interface. We have completed
* A GPUArray structure and backend that allows a SML user to create and manipulate arrays on the GPU, and allows 
  interfacing with the SML builtin Array structure, so a user can copy to and from SML arrays. Additionally, this
  structure allows a user to write thier own CUDA kernels, and then call these kernels on the GPUArrays. 
* A fast serial Sequence implementation to benchmark against. 
* Testing frameworks to assure the interface between SML and CUDA (memory copying, kernel launches) are functioning properly. 
* Begun implementation of Sequence primitives for integers in CUDA. So far we have implemented tabulate, map, reduce, and scan. 

# Goals and Deliverables

# Preliminary Results

# Updated Schedule

# Issues
