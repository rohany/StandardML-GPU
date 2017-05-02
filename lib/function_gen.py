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
  orig_fun_name = first_line[start:end]
  function_name = orig_fun_name + arg_type
  lines[0] = lines[0][:end] + arg_type + lines[0][end:]
  out.extend(lines)
  out.append("\n")
  dev_name = function_name + "_dev"
  dev_ptr = "__device__ " + fun_type + " " + dev_name + " = " + function_name + ";\n\n"
  out.append(dev_ptr)
  out.append("extern \"C\"\n")
  gen_name = "gen_" + function_name
  out.append("void* gen_"+function_name+"(){\n\n")
  out.append("\t" + fun_type + " local;\n\n")
  copy_line = "\tcudaMemcpyFromSymbol(&local, " + dev_name + ", sizeof(" + fun_type + "));\n\n"
  out.append(copy_line)
  out.append("\treturn (void*)local;\n\n")
  out.append("}\n\n")
  return (out, orig_fun_name, gen_name)


(out, orig_fun_name, gen_name) = gen_output(input_fun)

out_type = arg_type[1:]

for i in out:
  output_file.write(i)


#print out_type

#print orig_fun_name
#print gen_name

sml_file_name = "userkernels/" + out_type + "lambdas.sml"


#print sml_file_name

sml_read_lines = open(sml_file_name, "r").readlines()
sml_read_lines = sml_read_lines[:len(sml_read_lines)-1]

import_string = "\tval gen_" + orig_fun_name + " = _import \"" + gen_name + "\" public : unit -> MLton.Pointer.t;\n"
gen_string = "\tval " + orig_fun_name + " = gen_" + orig_fun_name + "()\n\n"

#print import_string
#print gen_string
sml_read_lines.append("\n")
sml_read_lines.append(import_string)
sml_read_lines.append(gen_string)
sml_read_lines.append("end\n")

write_file = open(sml_file_name, "w")
for i in sml_read_lines:
  write_file.write(i)

#print sml_read_lines
