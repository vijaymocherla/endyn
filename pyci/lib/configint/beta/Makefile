CC=gcc
CFLAGS=-fPIC -O3
LDFLAGS=-shared -lm

librcisd.so: rcisd.h rcisd.c
	$$(CC) -o librcisd.so $(LDFLAGS) $(CFLAGS) rcisd.c

clean:
	rm -f librcisd.so rcisd.o
