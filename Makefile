all: main

test: main.c deepcl.h
	gcc -O0 main.c

main: main.c deepcl.h
	gcc -O2 main.c

debug: main.c deepcl.h
	gcc -g main.c

check: a.out
	valgrind --leak-check=full ./a.out