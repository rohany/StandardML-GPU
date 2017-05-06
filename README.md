# StandardML-GPU
Adding fast CUDA bindings to Standard ML

For more information, check out our website at https://rohany.github.io/StandardML-GPU

To see SML-GPU in action, clone the repository with `git clone https://github.com/rohany/StandardML-GPU/`.

Install the dependencies 
~~~
sudo apt-get install mlton
~~~
Follow the Nvidia CUDA setup guide to install CUDA. Then head over to the tests directory, and make sure
that cuda path in the Makefiles is the same as the path to your installation. 

Then just run make, './<executable> <input size>`, and see the speedups!
