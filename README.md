# StandardML-GPU
Adding fast CUDA bindings to Standard ML

For more information, check out our website at https://rohany.github.io/StandardML-GPU

For quick usage, compile your .mlb file with the following : 

~~~
mlton -default-ann 'allowFFI true' -link-opt 'rdynamic -L/<path to your cuda install>' -lcudart -lstdc++' .mlb file <path to library gpublob.o>
~~~

