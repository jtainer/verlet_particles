// 
// Wrapper for CUDA functions
//
// 2022, Jonathan Tainer
//

#ifndef PHYSICS_H
#define PHYSICS_H

#include "pointmass.h"

PointMass* createElements(int numOfElements);

void deleteElements(PointMass* element);

void copyToDev(PointMass* devElement, PointMass* sysElement, int numOfElements);

void copyToSys(PointMass* sysElement, PointMass* devElement, int numOfElements);

void step(PointMass* element, int numOfElements, float dt);

#endif
