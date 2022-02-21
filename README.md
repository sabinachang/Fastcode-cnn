# Fast-Code-CNN-Kernel

## Machine
The project code was test ran on the department ece machine 005. 

## Our Implementation
The kernel_driver.c file initializes the input data, invokes the three kernel layers, and reports the performance results. For all three layers' kernel and packing procedure, they are stored in separate .h files. To invoke the kernel driver, simlpy move to root directory of the project and run make. 

## Baseline
Inside reference/ you could find the baseline project.  To test run the baseline, move to reference/simple_cnn-master/ and run make. We commented out the forth layer and hardcode the input matrix to check against our implementation correctness (so current baseline error is high). The input matrix is set at line 208-209 in example1.cpp. You could uncomment the forth layer and remove the hardcode where input is set to see the complete behavior of the baseline. 