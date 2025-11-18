#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
using namespace std;
#define TOTAL_NODE 4
#define TOTAL_ANT 30

struct edges{
    float cost;
    float pheromone;
};

__global__ void travel(int totalNodes,int totalAnts, edges* map, float* delta){
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    bool visited[TOTAL_NODE];
    bool temp[TOTAL_NODE][TOTAL_NODE]; // Boolean matrix to store edges that kth ant went through
    bool nextNodeExist = false;

    // Random 
    curandState_t state;
    curand_init(gidx, /* the seed controls the sequence of random values that are produced */
    0, /* the sequence number is only important with multiple cores */
    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
    &state);
    
    // Inititalize temp
    for(int i = 0; i < totalNodes; i++){
        for(int j = 0; j < totalNodes; j++){
            temp[i][j] = false;
        }
    }

    float totalCost = 0; // Lk
    int currNode = 0;
    int nextNode = 1;

    // Inititalize visited array for each ant
    for(int i = 0; i < totalNodes; i++) visited[i] = 0;
    visited[0] = true;

    // Ant kth travel thorugh every node
    if(gidx < totalAnts){
        for(int i = 0; i < totalNodes - 1; i++){
            // Calculate for next node
            // Getting totalProbs
            nextNodeExist = false;
            float probs[TOTAL_NODE];
            float totalProbs = 0;
            for(int l = 0; l < totalNodes; l++) probs[l] = 0;

            for(int j = 0; j < totalNodes; j++){
                if(!visited[j] && map[currNode*totalNodes+j].cost > 0){ // connects
                    nextNodeExist = true; // probs is not all 0
                    float Tij = map[currNode*totalNodes+j].pheromone;
                    float nij = 1.0 / map[currNode*totalNodes+j].cost; 
                    probs[j] = Tij * nij; 
                    totalProbs += Tij * nij;
                }
            }

            // If no connecting nodes to go, stop
            if(!nextNodeExist) break;

            // Getting all probabilities
            for(int k = 0; k < totalNodes; k++){
                if(probs[k] > 0){
                    probs[k] = probs[k] / totalProbs;
                }
            }

            // Turning probs to cumulative sum
            for(int a = 0; a < totalNodes; a++){
                if(a == 0){
                    probs[a] = 1 - probs[a];
                }
                else{
                    probs[a] = probs[a-1] - probs[a];
                }
            }

            float r = curand_uniform(&state);

            // Random next node
            for(int b = 0; b < totalNodes; b++){
                if(r >= probs[b]){
                    nextNode = b;
                    break;
                }
            }

            // Mark edge as visited
            totalCost += map[currNode*totalNodes + nextNode].cost;
            temp[currNode][nextNode] = true;
            temp[nextNode][currNode] = true;
            currNode = nextNode;
            visited[currNode] = true;
        }
        
        // Add from finish to start only if start and finish connects
        if(nextNodeExist && map[currNode*totalNodes].cost > 0){
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
}

__global__ void updatePheromones(int totalNodes, float evaRate, edges* map, float* delta){
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int gidx = row*totalNodes + col;
    
    if(col < totalNodes && row < totalNodes){
        map[gidx].pheromone = (1-evaRate) * map[gidx].pheromone + delta[gidx];
    }
}

void ACOParallel(int totalNodes, int totalAnts, float evaRate, size_t mapSize, edges* map, int epochs){
    edges* d_map;
    float* delta; // Pheromone delta matrix
    dim3 dimGrid(2,2,1);
    dim3 dimBlock(2,2,1);

    // Allocate for device map and delta matrix
    cudaMalloc((void**) &d_map, mapSize);
    cudaMalloc((void**) &delta, totalNodes*totalNodes*sizeof(float));

    // Copy host to device and travel
    cudaMemcpy(d_map, map, mapSize, cudaMemcpyHostToDevice);
    for(int i = 0; i < epochs; i++){
        cudaMemset((void*) delta, 0, sizeof(float)*totalNodes*totalNodes);
        travel<<<1, totalAnts>>>(totalNodes, totalAnts, d_map, delta);
        cudaDeviceSynchronize();
        updatePheromones<<<dimGrid, dimBlock>>>(totalNodes, evaRate, d_map, delta);
        cudaDeviceSynchronize();
        cout << "Done epoch " << i << endl;
    }

    // Copy device to host
    cudaMemcpy(map, d_map, mapSize, cudaMemcpyDeviceToHost);
    cudaFree(d_map);
}

void init(int totalNodes,edges* map, string fileNameCost, string fileNamePheromone){
    ifstream fileReader;
    ifstream fileReader2;

    // Read cost
    fileReader.open(fileNameCost);

    for(int i = 0; i < totalNodes; i++){
        for(int j = 0; j < totalNodes; j++){
           fileReader >> map[i*totalNodes+j].cost; 
        }
    }

    // Read pheromone
    fileReader2.open(fileNamePheromone);
    for(int i = 0; i < totalNodes; i++){
        for(int j = 0; j < totalNodes; j++){
           fileReader2 >> map[i*totalNodes+j].pheromone; 
        }
    }
}

void printCost(int totalNodes, edges* map){
    cout << "---------- COST PRINTING -----------" << endl;
    for(int i = 0; i < totalNodes; i++){
        for(int j = 0; j < totalNodes; j++){
            cout << map[i*totalNodes + j].cost << " ";
        }
        cout << endl;
    }
}

void printPheromone(int totalNodes, edges* map){
    cout << "---------- PHEROMONE PRINTING -----------" << endl;
    for(int i = 0; i < totalNodes; i++){
        for(int j = 0; j < totalNodes; j++){
            cout << map[i*totalNodes + j].pheromone << " ";
        }
        cout << endl;
    }
}

int main(){
    int totalNodes = TOTAL_NODE;
    int totalAnts = TOTAL_ANT;
    float evaRate = 0.5;
    size_t mapSize = sizeof(edges)*totalNodes*totalNodes;
    edges* map = (edges*)malloc(mapSize);
    string costFile = "cost.txt";
    string pheromoneFile = "pheromone.txt";
    int epochs = 1;

    init(totalNodes, map, "cost.txt","pheromone.txt");
    cout << "AFTER INIT" << endl;
    printCost(totalNodes, map);
    printPheromone(totalNodes, map);
    ACOParallel(totalNodes, totalAnts, evaRate, mapSize, map, epochs);
    printPheromone(totalNodes, map);
    free(map);
}