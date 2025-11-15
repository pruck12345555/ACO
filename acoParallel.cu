#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
#define TOTAL_NODE 20
#define TOTAL_ANT 30

struct edges{
    float cost;
    float pheromone;
};

__global__ void travel(int totalNodes, edges* map, float* delta){
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    bool visited[TOTAL_NODE];
    bool temp[TOTAL_NODE][TOTAL_NODE]; // Boolean matrix to store edges that kth ant went through

    // Inititalize temp
    for(int i = 0; i < TOTAL_NODE; i++){
        for(int j = 0; j < TOTAL_NODE; j++){
            temp[i][j] = 0;
        }
    }

    float totalCost = 0; // Lk
    int currNode = 0;
    int nextNode = 1;

    // Inititalize visited array for each ant
    for(int i = 0; i < totalNodes; i++) visited[i] = 0;
    visited[0] = true;

    // Ant kth travel thorugh every node
    if(gidx < TOTAL_ANT){
        for(int i = 0; i < totalNodes - 1; i++){
            // Calculate for next node
            // Getting totalProbs
            float probs[TOTAL_NODE];
            float totalProbs = 0;
            for(int l = 0; l < totalNodes; l++) probs[l] = 0;

            for(int j = 0; j < totalNodes; j++){
                if(!visited[j] && map[currNode*totalNodes+j].cost > 0){ // connects
                    float Tij = map[currNode*totalNodes+j].pheromone;
                    float nij = 1.0 / map[currNode*totalNodes+j].cost; 
                    probs[j] = Tij * nij; 
                    totalProbs += Tij * nij;
                }
            }

            // Getting all probabilities
            for(int k = 0; k < totalNodes; k++){
                if(probs[k] > 0){
                    probs[k] = probs[k] / totalProbs;
                }
            }

            // Turning probs to cumulative sum


            // Random and choose node
            

            // Get next node
            totalCost += map[currNode*totalNodes + nextNode].cost;
            temp[currNode][nextNode] = true;
            temp[nextNode][currNode] = true;
            currNode = nextNode;
            visited[currNode] = true;
        }
        
        // Add from finish to start
        totalCost += map[currNode*totalNodes].cost;
        temp[currNode][0] = true;
        temp[0][currNode] = true;

        // Ant kth finished travel through every node (got Lk)
        // add to pheromone delta matrix
        for(int i= 0; i < totalNodes; i++){
            for(int j = 0; j < totalNodes; j++){
                // If passed, update
                if(temp[i][j]){
                    atomicAdd(&delta[i*totalNodes+j], 1.0 / totalCost);
                }
            }
        }
    }  
}

__global__ void updatePheromones(int totalNodes, float evaRate, edges* map, float* delta){
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int gidx = row*totalNodes + col;
    
    if(col < TOTAL_NODE && row < TOTAL_NODE){
        map[gidx].pheromone = (1-evaRate) * map[gidx].pheromone + delta[gidx];
    }
}

void ACOParallel(int totalNodes, float evaRate, size_t mapSize, edges* map, int epochs){
    edges* d_map;
    float* delta; // Pheromone delta matrix
    int epoch = 1;
    dim3 dimGrid(2,2,1);
    dim3 dimBlock(10,10,1);

    // Allocate for device map and delta matrix
    cudaMalloc((void**) &d_map, mapSize);
    cudaMalloc((void**) &delta, totalNodes*totalNodes*sizeof(float));

    // Copy host to device and travel
    cudaMemcpy(d_map, map, mapSize, cudaMemcpyHostToDevice);
    for(int i = 0; i < epochs; i++){
        cudaMemset((void*) delta, 0, sizeof(float)*totalNodes*totalNodes);
        travel<<<1, TOTAL_ANT>>>(totalNodes, d_map, delta);
        updatePheromones<<<dimGrid, dimBlock>>>(totalNodes, evaRate, d_map, delta);
    }

    // Copy device to host
    cudaMemcpy(map, d_map, mapSize, cudaMemcpyDeviceToHost);
    cudaFree(d_map);
}

int main(){
    int totalNodes = TOTAL_NODE;
    size_t mapSize = sizeof(edges)*totalNodes*totalNodes;
    edges* map = (edges*)malloc(mapSize);

}