// 
// Gravitational particle system
//
// 2022, Jonathan Tainer
//

#include "physics.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <raylib.h>

const float stepSize = 0.01f;
const int stepsPerFrame = 4;

int main() {

	srand(time(NULL));

	const int numOfElements = 2048;

	// Create particle system
	PointMass* sysptr = (PointMass*)malloc(sizeof(PointMass) * numOfElements);
	Color* color = (Color*)malloc(sizeof(Color) * numOfElements);
	for (int i = 0; i < numOfElements; i++) {
		sysptr[i].pos = (vec2) { (float) (rand() % 1920), (float) (rand() % 1080) };
		sysptr[i].vel = (vec2) { (float) (rand() % 5) - 2, (float) (rand() % 5) - 2 };
		sysptr[i].acc0 = (vec2) { 0.f, 0.f };
		sysptr[i].acc1 = (vec2) { 0.f, 0.f };
		sysptr[i].rad = (float) ((rand() % 10) + 10);
		sysptr[i].mass = 75.f;
		
		color[i] = (Color) { (unsigned char) (rand() % 256), (unsigned char) (rand() % 256), (unsigned char) (rand() % 256), 255 };
	}

	// Make one big particle that follows the mouse
	sysptr[0].rad = 100.f;

	PointMass* devptr = createElements(numOfElements);
	if (devptr == nullptr) return 1;
	copyToDev(devptr, sysptr, numOfElements);

	// Raylib setup
	SetConfigFlags(FLAG_WINDOW_ALWAYS_RUN);
	InitWindow(1920, 1080, "Particle Sim");
	ToggleFullscreen();

	while (!WindowShouldClose()) {
		
		BeginDrawing();
		ClearBackground(RAYWHITE);
		
		for (int i = 0; i < numOfElements; i++) {
			DrawCircle(sysptr[i].pos.x, sysptr[i].pos.y, sysptr[i].rad, color[i]);
		}
		
		DrawFPS(10, 10);
		EndDrawing();

		// Compute next state of particle system
		sysptr[0].pos = { GetMousePosition().x, GetMousePosition().y };
		sysptr[0].vel = { 0.f, 0.f };
		copyToDev(devptr, sysptr, numOfElements);
		for (int i = 0; i < stepsPerFrame; i++)
			step(devptr, numOfElements, stepSize);
		copyToSys(sysptr, devptr, numOfElements);
		sysptr[0].pos = { GetMousePosition().x, GetMousePosition().y };
	}

	CloseWindow();
	deleteElements(devptr);
	free(sysptr);
	free(color);
	return 0;
}
