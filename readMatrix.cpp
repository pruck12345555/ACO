#include <iostream>
#include <fstream>
using namespace std;
#define NUM_NODES 4

struct edges
{
    float pheromone;
    float cost;
};


void getCost(edges* map, string fileName){
    ifstream fileReader;
    fileReader.open(fileName);

    for(int i = 0; i < NUM_NODES; i++){
        for(int j = 0; j < NUM_NODES; j++){
           fileReader >> map[i*NUM_NODES+j].cost; 
        }
    }

}

void getPheromone(edges* map, string fileName){
    ifstream fileReader;
    fileReader.open(fileName);

    for(int i = 0; i < NUM_NODES; i++){
        for(int j = 0; j < NUM_NODES; j++){
           fileReader >> map[i*NUM_NODES+j].pheromone; 
        }
    }
}

void printCost(edges* map){
    cout << "---------- COST PRINTING -----------" << endl;
    for(int i = 0; i < NUM_NODES; i++){
        for(int j = 0; j < NUM_NODES; j++){
            cout << map[i*NUM_NODES + j].cost << " ";
        }
        cout << endl;
    }
}

void printPheromone(edges* map){
    cout << "---------- PHEROMONE PRINTING -----------" << endl;
    for(int i = 0; i < NUM_NODES; i++){
        for(int j = 0; j < NUM_NODES; j++){
            cout << map[i*NUM_NODES + j].pheromone << " ";
        }
        cout << endl;
    }
}

int main(){
    edges* map = (edges*)malloc(sizeof(edges)*NUM_NODES*NUM_NODES); 
    getCost(map, "cost.txt");
    getPheromone(map, "pheromone.txt");
    printCost(map);
    printPheromone(map);
    return 0;
}