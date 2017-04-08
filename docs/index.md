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
(* add more here lol *)

# Resources
We will be starting from scratch. We would like the course staff to install the MLton compiler on the Gates cluster machines - 
can be done with 'apt-get install mlton'.

# Goals and Deliverables

# Platform Choice
Since functional programs naturally express a very data-parallel method of computation, a GPU is a natural platform to 
run the code on. Additionally, current research is going on in running this code on multi-core CPU's, so it would be interesting
to compare our implementation with that research. 

# Schedule

same





You can use the [editor on GitHub](https://github.com/rohany/StandardML-GPU/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/rohany/StandardML-GPU/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
