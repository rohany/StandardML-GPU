#!/bin/python
import sys

if(len(sys.argv) < 4):
  print "Usage : function_gen <fun_type> <input_file> <output_file>"

fun_type = sys.argv[1]
arg_type = fun_type[(len(fun_type)-4):(len(fun_type))]
input_fun = open(sys.argv[2], 'r')
output_file = open(sys.argv[3], 'a+')

def gen_output(in_file):
  out = []
  out.append("__device__\n")
  lines = in_file.readlines()
  first_line = lines[0]
  start = first_line.index(" ") + 1
  end = first_line.index("(")
  function_name = first_line[start:end] + arg_type
  lines[0] = lines[0][:end] + arg_type + lines[0][end:]
  out.extend(lines)
  out.append("\n")
  dev_name = function_name + "_dev"
  dev_ptr = "__device__ " + fun_type + " " + dev_name + " = " + function_name + ";\n\n"
  out.append(dev_ptr)
  out.append("extern \"C\"\n")
  out.append("void* gen_"+function_name+"(){\n\n")
  out.append("\t" + fun_type + " local;\n\n")
  copy_line = "\tcudaMemcpyFromSymbol(&local, " + dev_name + ", sizeof(" + fun_type + "));\n\n"
  out.append(copy_line)
  out.append("\treturn (void*)local;\n\n")
  out.append("}\n\n")
  return out


out = gen_output(input_fun)

for i in out:
  output_file.write(i)
