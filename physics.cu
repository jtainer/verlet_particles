// 
// Wrapper for CUDA functions
// 
// 2022, Jonathan Tainer
// 

#include <math.h>
#include "kernel.h"
#include "physics.h"

PointMass* createElements(int numOfElements) {
	PointMass* ptr = nullptr;
	cudaMalloc((void**)&ptr, sizeof(PointMass) * numOfElements);
	return ptr;
}

void deleteElements(PointMass* element) {
	cudaFree(element);
}

void copyToDev(PointMass* devElement, PointMass* sysElement, int numOfElements) {
	cudaMemcpy(devElement, sysElement, sizeof(PointMass) * numOfElements, cudaMemcpyHostToDevice);
}

void copyToSys(PointMass* sysElement, PointMass* devElement, int numOfElements) {
	cudaMemcpy(sysElement, devElement, sizeof(PointMass) * numOfElements, cudaMemcpyDeviceToHost);
}

void step(PointMass* element, int numOfElements, float dt) {
	verletUpdatePos<<<numOfElements / 512, 512>>>(element, numOfElements, dt);
	verletUpdateAcc<<<numOfElements / 512, 512>>>(element, numOfElements);
	verletUpdateVel<<<numOfElements / 512, 512>>>(element, numOfElements, dt);
}


