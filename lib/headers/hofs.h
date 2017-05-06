#ifndef HOFS_H
#define HOFS_H
typedef int (*tabulate_fun_int)(int x);
typedef int (*map_fun_int)(int x);
typedef int (*reduce_fun_int)(int x, int y);
typedef int (*scan_fun_int)(int x, int y);
typedef bool (*filter_fun_int)(int x);
typedef int (*zipwith_fun_int)(int x, int y);

typedef float (*tabulate_fun_float)(int x);
typedef float (*map_fun_float)(float x);
typedef float (*reduce_fun_float)(float x, float y);
typedef float (*scan_fun_float)(float x, float y);
typedef bool (*filter_fun_float)(float x);
typedef float (*zipwith_fun_float)(float x, float y);


typedef int2 (*tabulate_fun_int_tuple)(int x);
typedef int2 (*map_fun_int_tuple)(int x_1, int x_2);
typedef int2 (*reduce_fun_int_tuple)(int x_1, int x_2, int y_1, int y_2);
typedef int2 (*scan_fun_int_tuple)(int x_1, int x_2, int y_1, int y_2);
typedef bool (*filter_fun_int_tuple)(int x_1, int x_2);
typedef int2 (*zipwith_fun_int_tuple)(int x_1, int x_2, int y_1, int y_2);
#endif
