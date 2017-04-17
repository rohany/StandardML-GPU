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
#endif
