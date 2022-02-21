all:
	gcc -Wall -mavx -mfma kernel_driver.c -o kernel_driver.x -O3 -std=c99 -march=native -fopenmp
	./kernel_driver.x 1 12

clean:
	rm -rf *~
	rm -rf *.x
