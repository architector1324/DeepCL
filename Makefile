all: basic

basic: ./examples/basic/main.c deepcl.h
	gcc -O2 -I ./ ./examples/basic/main.c -o ./bin/basic

basic_debug: ./examples/basic/main.c deepcl.h
	gcc -g  -I ./ ./examples/basic/main.c -o ./bin/basic

basic_check: a.out
	valgrind --leak-check=full ./bin/basic