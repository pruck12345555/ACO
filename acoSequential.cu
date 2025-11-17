#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
using namespace std;
#define TOTAL_NODE 4

struct edges{
    float cost;
    float pheromone;
};

bool myComparison(const pair<float,int> &a,const pair<float,int> &b)
{
    return a.first>b.first;
}

class ACO{ // Sequential code
public:
    int totalNodes;
    int totalAnts;
    float evaRate;
    int epochs;
    //vector<vector<edges>> map;
    edges map[TOTAL_NODE][TOTAL_NODE];
    float delta[TOTAL_NODE][TOTAL_NODE];

    ACO(int totalNodes, int totalAnts, float evaRate, int epochs){
        this->totalAnts = totalAnts;
        this->evaRate = evaRate;
        this->epochs = epochs;
        this->totalNodes = totalNodes;
    }

    void printPhems(){
        for(int i = 0; i< totalNodes; i++){
            for(int j = 0; j < totalNodes; j++){
                cout << map[i][j].pheromone << " ";
            }
            cout << endl;
        }
    }

    void init(string fileNameCost, string fileNamePheromone){
        ifstream fileReader;
        ifstream fileReader2;

        // Read cost
        fileReader.open(fileNameCost);
        for(int i = 0; i < totalNodes; i++){
            for(int j = 0; j < totalNodes; j++){
                fileReader >> map[i][j].cost; 
            }
        }
        fileReader.close();

        // Read pheromone
        fileReader2.open(fileNamePheromone);
        for(int i = 0; i < totalNodes; i++){
            for(int j = 0; j < totalNodes; j++){
            fileReader2 >> map[i][j].pheromone; 
            }
        }
        fileReader2.close();

        // Init delta
        for(int i = 0; i< totalNodes; i++){
            for(int j = 0; j < totalNodes; j++){
                delta[i][j] = 0;
            }
        }
    }

    void start(){
        init("cost.txt", "pheromone.txt");
        cout << "FIRST READ" << endl;
        printPhems();
        for(int i = 0; i < epochs; i++){
            travel();
            updatePheromones();
            cout << "DONE EPOCH " << i << endl;
            printPhems();
        }
    }

    void travel(){
        // Loop for each ant
        for(int i = 0; i < totalAnts; i++){
            // current node (starts at node 0)
            int currNode = 0;
            bool nextNodeExist = true;
            // visited nodes
            vector<bool> visited(totalNodes, 0);
            float totalCost = 0; // Lk
            vector<pair<int, int>> traveledThrough;
            visited[0] = true;

            // loop until traveled every node
            for(int j = 0; j < totalNodes-1; j++){
                nextNodeExist = true;
                int nextNode = getNextNode(visited, currNode);
                if(nextNode < 0){
                    nextNodeExist = false;
                    break;
                } 
                totalCost += map[currNode][nextNode].cost;
                pair<int, int> nodePair;
                nodePair.first = currNode;
                nodePair.second = nextNode;
                traveledThrough.push_back(nodePair);
                currNode = nextNode;
                visited[currNode] = true;          
            }

            // Loop succeed & Last node connects with first
            if(nextNodeExist && map[currNode][0].cost > 0){
                totalCost += map[currNode][0].cost; // From last node to start

                // Plus on delta map for every node that passes through
                for(int k = 0; k < traveledThrough.size(); k++){
                    delta[traveledThrough[k].first][traveledThrough[k].second] += 1.0 / totalCost;
                    delta[traveledThrough[k].second][traveledThrough[k].first] += 1.0 / totalCost;
                }
            }
            
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

        if(probs.empty()) return -1;

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
    int totalNodes = TOTAL_NODE;
    int totalAnts = 200;
    float evaRate = 0.5;
    int epochs = 200;
    ACO aco(totalNodes, totalAnts, evaRate, epochs);
    aco.start();

    return 0;
}