// 
// CUDA kernels to compute the motion of a system of particles
//
// 2022, Jonathan Tainer
//

#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include "pointmass.h"

__global__
void verletUpdatePos(PointMass* element, int numOfElements, float dt);

__global__
void verletUpdateAcc(PointMass* element, int numOfElements);

__global__
void verletUpdateVel(PointMass* element, int numOfElements, float dt);

#endif
