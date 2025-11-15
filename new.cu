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

bool myComparison(const pair<float,int> &a,const pair<float,int> &b)
{
    return a.first>b.first;
}

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

class ACOParallel{ // Parallel code
    // Host attributes
    int totalNodes;
    int totalAnt;
    float evaRate;
    edges *map = (edges*)malloc(sizeof(edges)*TOTAL_NODE*TOTAL_NODE);

    // Device attributes
    edges* d_map;
    float* d_delta;

    void start(){
        initialize();
        const int NUM_BLOCK = 2;
        const int NUM_GRID = TOTAL_NODE / NUM_BLOCK;
        dim3 dimGridAll(NUM_GRID, NUM_GRID, 1);
        dim3 dimBlockAll(NUM_BLOCK, NUM_BLOCK, 1);
        travel<<<dimGridAll,dimBlockAll>>>();
    }

    void initialize(){
        cudaMalloc((void**) &d_map, sizeof(edges)*TOTAL_NODE*TOTAL_NODE);
        cudaMalloc((void**) &d_delta, sizeof(float)*TOTAL_NODE*TOTAL_NODE);
    }

    __global__ void travel(){
        // 1 thread = 1 ant (1.)
        int gidx = threadIdx.x + blockDim.x * blockIdx.x;
        int currNode = 0;
        bool visited[TOTAL_NODE]; // visited array for each ant
        float totalCost = 0; // total cost for each ant (Lk)
        bool traveledThrough[TOTAL_NODE*TOTAL_NODE]; // traveled through edges for each trip

        // Initialize visited
        if(gidx < totalNodes) visited[gidx] = 0;

        // Every ant goes through every node (2.)
        for(int j = 0; j < totalNodes-1; j++){
            int nextNode = getNextNode(visited, currNode);
            totalCost += map[currNode*totalNodes+nextNode].cost;
            traveledThrough[currNode*totalNodes+nextNode] = true;
            traveledThrough[nextNode*totalNodes+currNode] = true;
            currNode = nextNode;
            visited[currNode] = true;    
        }

        totalCost += map[0+currNode].cost; // From last node back to start

        // Plus on delta map for every node that passes through (3.)
        if(gidx < totalNodes){
            for(int i = 0; i< totalNodes; i++){
                if(traveledThrough[gidx*totalNodes + i]){
                    atomicAdd(&d_delta[gidx*totalNodes+i],1.0/totalCost);
                }
            }
        }
        __syncthreads();

        // Update pheromones after delta map is done (one time)
        if(gidx == 0) updatePheromone();
    }

    // get denominator (4.)
    __global__ void getTotalProbs(float* totalProb, int currNode, bool visited[], int *totalConnect){
        int gidx = blockDim.x * blockIdx.x + threadIdx.x;
        if(gidx < totalNodes){
            if(!visited[gidx] && map[currNode*totalNodes + gidx].cost > 0){
                *totalProb += map[currNode*totalNodes + gidx].pheromone * (1.0 / map[currNode*totalNodes + gidx].cost);
                *totalConnect++;
            }
        }
    }

    // 5.
    __global__ void getProbs(float *probs, bool visited[], int currNode, float totalProb){
        int gidx = blockDim.x * blockIdx.x + threadIdx.x;

        if(gidx < totalNodes){
            if(!visited[gidx] && map[currNode*totalNodes + gidx].cost > 0){
                float currProb = map[currNode*totalNodes + gidx].pheromone * (1.0 / map[currNode*totalNodes + gidx].cost) / totalProb;
                probs[gidx] = currProb;
            }
        }
    }

    __global__ int getNextNode(bool visited[], int currNode){
        // Find max prob. and choose next node
        float r = ((float) rand() / (RAND_MAX)); // 0 <= r <= 1
        //vector<pair<float,int>> probs; // first = probs, second = node index
        float probs[TOTAL_NODE];
        float totalProb = 0;
        int totalConnect = 0;
        int nextNode;

        // Get denominator
        getTotalProbs(&totalProb, currNode, visited, &totalConnect);

        // Get all probs of connected nodes
        getProbs(probs, visited, currNode, totalProb);

        // easy random return
        for(int i = 0; i < totalNodes - 1; i++){
            if(probs[i] <= r) return i;
        }

        // return last index if loop does not return
        // total connected nodes needed
        return probs[totalConnect-1];
    }

    // 7.
    __global__ void updatePheromone(){
        // CANNOT USE BLOCKDIM Y
        const int col = blockDim.x * blockIdx.x + threadIdx.x;
        const int row = blockDim.y * blockIdx.y + threadIdx.y;

        if(row < totalNodes && col < totalNodes){
            map[row * totalNodes + col].pheromone = ((1 - evaRate) * map[row * totalNodes + col].pheromone) + d_delta[row * totalNodes + col];
        }
    }
};

class ACO{ // Sequential code
    int totalNodes;
    int totalAnts;
    float evaRate = 0.5;
    vector<vector<edges>> map;
    vector<vector<float>> delta;

    void travel(){
        // Loop for each ant
        for(int i = 0; i < totalAnts; i++){
            // current node (starts at node 0)
            int currNode = 0;
            // visited nodes
            vector<bool> visited(totalNodes, 0);
            float totalCost = 0; // Lk
            vector<pair<int, int>> traveledThrough;
            visited[0] = true;

            // loop until traveled every node
            for(int j = 0; j < totalNodes-1; j++){
                int nextNode = getNextNode(visited, currNode);
                totalCost += map[currNode][nextNode].cost;
                pair<int, int> nodePair;
                nodePair.first = currNode;
                nodePair.second = nextNode;
                traveledThrough.push_back(nodePair);
                currNode = nextNode;
                visited[currNode] = true;          
            }

            totalCost += map[currNode][0].cost; // From last node to start

            // Plus on delta map for every node that passes through
            for(int k = 0; k < traveledThrough.size(); k++){
                delta[traveledThrough[k].first][traveledThrough[k].second] += 1.0 / totalCost;
                delta[traveledThrough[k].second][traveledThrough[k].first] += 1.0 / totalCost;
            }
            updatePheromones();
        }
    }

    int getNextNode(vector<bool> visited, int currNode){ 
        // Find max prob. and choose next node
        float r = ((float) rand() / (RAND_MAX)); // 0 <= r <= 1
        vector<pair<float,int>> probs; // first = probs, second = node index
        float totalProb = 0;
        int nextNode;

        // Get denominator
        for(int i = 0;  i < totalNodes; i++){
            if(!visited[i] && map[currNode][i].cost > 0){
                totalProb += map[currNode][i].pheromone * (1.0 / map[currNode][i].cost);
            }
        }

        // Get all probs of connected nodes
        for(int i = 0; i< totalNodes; i++){
            if(!visited[i] && map[currNode][i].cost > 0){
                float currProb = map[currNode][i].pheromone * (1.0 / map[currNode][i].cost) / totalProb;
                pair<float, int> probWithIndex;
                probWithIndex.first = currProb;
                probWithIndex.second = i;
                probs.push_back(probWithIndex);
            }
        }

        // Sort the probabilistic vector
        sort(probs.begin(), probs.end(), myComparison);

        float cum = 1;
        // Loop cumulative sum
        for(int i = 0; i < probs.size()-1; i++){
            float nextCum = cum - probs[i].first;
            if(r > nextCum && r <= cum) return probs[i].second; // If in the range, return index
            cum = nextCum;
        }

        // return last index if loop does not return
        return probs[probs.size()-1].second;
    }

    // update pheromone (every update old pheromone evaporates by evaRate)
    void updatePheromones(){
        for(int i = 0; i < totalNodes; i++){
            for(int j = 0; j < totalNodes; j++){
                float oldPheromone = map[i][j].pheromone;
                map[i][j].pheromone = ((1-evaRate) * oldPheromone) + delta[i][j];
            }
        }
    }
};

int main(){
    long long num1 = 184465;
    long long num2 = 223245;
    cout << num1*num2 << endl;
    return 0;
}