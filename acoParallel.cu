#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
#define TOTAL_NODE 20

struct edges{
    float cost;
    float pheromone;
};

//loops
// ----------- เดินทาง ---------------
// 1. loop มดทุกตัว ---> thread ละตัว ?
// 2. loop มดหนึ่งตัว เดินทางไปทุก node (บังคับ sequential)
// 3. loop เพิ่มค่า delta pheromone ในแต่ละ node ที่ผ่าน --> addDelta
// ----------- หา next node -----------
// 4. loop เอา denominator --> getTotalProb
// 5. loop เอา probability ของทุก node ที่เชื่อมกับ node ปัจจุบัน --> getProbs
// 6. loop cumulative sum เพื่อสุ่ม
// ------------ update pheromone -----------
// 7. loop อัปเดต pheromone จากตาราง delta --> matrix addition (with evaporation)

__global__ void travel(int totalNodes, edges* map, float* delta, int epoch){
    
}

void ACOParallel(int totalNodes, size_t mapSize, edges* map){
    edges* d_map;
    float* delta; // Pheromone delta matrix
    int epoch = 1;

    // Allocate for device map and delta matrix
    cudaMalloc((void**) d_map, mapSize);
    cudaMalloc((void**) delta, totalNodes*totalNodes*sizeof(float));

    // Copy host to device
    cudaMemcpy(d_map, map, mapSize, cudaMemcpyHostToDevice);
    travel(totalNodes, map, delta, epoch);

    // Copy device to host
    cudaMemcpy(map, d_map, mapSize, cudaMemcpyDeviceToHost);
    cudaFree(d_map);
}

int main(){
    int totalNodes = TOTAL_NODE;
    size_t mapSize = sizeof(edges)*totalNodes*totalNodes;
    edges* map = (edges*)malloc(mapSize);

}