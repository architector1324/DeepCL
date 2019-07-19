all: basic

basic: ./examples/basic/main.c deepio.h
	gcc -O2 -I ./ ./examples/basic/main.c -o ./bin/basic

basic_debug: ./examples/basic/main.c deepio.h
	gcc -g  -I ./ ./examples/basic/main.c -o ./bin/basic

basic_check: ./bin/basic
	valgrind --leak-check=full ./bin/basic