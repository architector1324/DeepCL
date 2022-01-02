all: simple

simple: simple_low simple_high

simple_low: ./examples/simple/low_level.c deepio.h
	gcc -O2 -I ./ ./examples/simple/low_level.c -o ./bin/simple_low

simple_low_debug: ./examples/simple/low_level.c deepio.h
	gcc -g  -I ./ ./examples/simple/low_level.c -o ./bin/simple_low

simple_low_check: ./bin/simple_low
	valgrind --leak-check=full ./bin/simple_low

simple_high: ./examples/simple/high_level.c deepio.h
	gcc -O2 -I ./ ./examples/simple/high_level.c -o ./bin/simple_high -lm

simple_high_debug: ./examples/simple/high_level.c deepio.h
	gcc -g  -I ./ ./examples/simple/high_level.c -o ./bin/simple_high -lm

simple_high_check: ./bin/simple_high
	valgrind --leak-check=full ./bin/simple_high
