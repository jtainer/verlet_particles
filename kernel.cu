// 
// CUDA kernels to compute motion of a system of particles
// 
// 2022, Jonathan Tainer
// 

#include "kernel.h"

#define G 100.f
#define R 50.f
#define K_PARTICLE 30000.f
#define K_WALL 5000.f
#define DRAG 50.f

__global__
void verletUpdatePos(PointMass* element, int numOfElements, float dt) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	if (tid < numOfElements) {
		vec2 v = element[tid].vel;
		vec2 a = element[tid].acc0;
		
		element[tid].pos.x += (v.x * dt) + (a.x * dt * dt);
		element[tid].pos.y += (v.y * dt) + (a.y * dt * dt);
	}
}

__global__
void verletUpdateAcc(PointMass* element, int numOfElements) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numOfElements) {
		
		// Reset acceleration vector to zero
		element[tid].acc0 = element[tid].acc1;
		element[tid].acc1 = (vec2) { 0.f, 0.f };

		// Compute elastic force of collisions with other particles
		for (int i = 0; i < numOfElements; i++) {
			if (i != tid) {
				
				// Detect if particles are colliding
				float dx = element[i].pos.x - element[tid].pos.x;
				float dy = element[i].pos.y - element[tid].pos.y;
				float distsqr = (dx * dx) + (dy * dy);
				float dist = sqrt(distsqr) + 0.1f;
				float overlap = element[tid].rad + element[i].rad - dist;

				float acc = 0.f;
				if (overlap > 0.f)
					acc = overlap * K_PARTICLE / element[tid].mass;
				
				float cosine = dx / dist;
				float sine = dy / dist;

				float accx = -acc * cosine;
				float accy = -acc * sine;

				// Add the computed acceleration component to the element's net acceleration
				element[tid].acc1.x += accx;
				element[tid].acc1.y += accy;
			}
		}

		// Compute elastic force of collisions with walls
		if (element[tid].pos.x < 0.f) {
			element[tid].acc1.x += (0.f - element[tid].pos.x) * K_WALL / element[tid].mass;
		}
		else if (element[tid].pos.x > 1920.f) {
			element[tid].acc1.x -= (element[tid].pos.x - 1920.f) * K_WALL / element[tid].mass;
		}
		
		if (element[tid].pos.y < 0.f) {
			element[tid].acc1.y += (0.f - element[tid].pos.y) * K_WALL / element[tid].mass;
		}
		else if (element[tid].pos.y > 1080.f) {
			element[tid].acc1.y -= (element[tid].pos.y - 1080.f) * K_WALL / element[tid].mass;
		}

		// Apply gravity
		element[tid].acc1.y += G;

		// Apply drag
		element[tid].acc1.x -= element[tid].vel.x * DRAG / element[tid].mass;
		element[tid].acc1.y -= element[tid].vel.y * DRAG / element[tid].mass;
	}
}

__global__
void verletUpdateVel(PointMass* element, int numOfElements, float dt) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	if (tid < numOfElements) {
		element[tid].vel.x += 0.5f * (element[tid].acc0.x + element[tid].acc1.x) * dt;
		element[tid].vel.y += 0.5f * (element[tid].acc0.y + element[tid].acc1.y) * dt;
	}
}

















