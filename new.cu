#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
#define TOTAL_NODE 250

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
    int totalNodes;
    int totalAnts;
    float evaRate = 0.5;
    int* map; //size = totalNodes * totalNodes

    __global__ void travel(){
        int gidx = blockIdx.x * blockDim.x + threadIdx.x; // หมายเลขของมดจากมดทั้งหมด

        // มด 1 ตัว = 1 THREAD
        if(gidx < totalAnts){
            int currNode = 0;
            bool visited[totalNodes]; // visited for each ant
            float totalCost = 0;
            int nextNode = getNextNode();
            vector<pair<int, int>> traveledThrough;
            
            // one ant travels every node
            for(int i = 0; i < totalNodes; i++){
               int nextNode = getNextNode();
               totalCost += map[currNode][nextNode]; // .cost
                pair<int, int> nodePair;
                nodePair.first = currNode;
                nodePair.second = nextNode;
                traveledThrough.push_back(nodePair);
                currNode = nextNode;
                visited[currNode] = true;
            }

            totalCost += map[currNode][0]; // .cost

            addDelta(totalCost);
        }
    }

    __global__ void addDelta(float totalCost, vector<pair<int, int>> traveledThrough){
        float toAdd = 1.0/totalCost;
        int size = traveledThrough.size();
        
        if(gidx < size){
            delta[traveledThrough[gidx].first][traveledThrough[gidx].second] += 1.0 / totalCost;
            delta[traveledThrough[gidx].first][traveledThrough[gidx].second] += 1.0 / totalCost;
        }
    }

    __global__ void getDenom(float totalProb, int currNode){
        int gidx = blockDim.x * blockIdx.x + threadIdx.x;
        if(!visited[gidx] && map[currNode][gidx].cost > 0){
            totalProb += map[currNode][gidx].pheromone * (1.0/map[currNode][gidx].cost);
        }
    }

    __global__ void getProbs(vector<pair<float, int>> probs, bool visited[], int currNode, float totalProb){
        int gidx = blockDim.x * blockIdx.x + threadIdx.x;
        if(!visited[gidx] && map[currNode][gidx].cost > 0){
            float currProb = map[currNode][gidx].pheromone * (1.0 / map[currNode][i].cost) / totalProb;
            pair<float, int> probWithIndex;
            probWithIndex.first = currProb;
            probWithIndex.second = i;
            probs.push_back(probWithIndex);
        }
    }

    int getNextNode(vector<bool> visited, int currNode){
        // Find max prob. and choose next node
        float r = ((float) rand() / (RAND_MAX));
        vector<pair<float,int>> probs;
        float totalProb = getDenom();
        int nextNode;

        getProbs(probs);
        
        //sort
        // loop cumulative sum
        
        return probs[probs.size() - 1].second;
    }

    void updatePheromone(){
        // matrix addition
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
        }

        updatePheromones();
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