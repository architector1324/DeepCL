all: main

main: main.c deepcl.h
	gcc -O2 main.c

debug: main.c deepcl.h
	gcc -g main.c