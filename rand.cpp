#include <iostream>
#include <random>
#include <vector>
#define TOTAL_NODE 100
using namespace std;

int main(){
    float cost[TOTAL_NODE][TOTAL_NODE];
    float phem[TOTAL_NODE][TOTAL_NODE];

    // Initialize random number generator
    default_random_engine generator;
    uniform_real_distribution<> dis(1.0, 100.0);
    // Fill cost and phem matrices with random float values
    for(int i = 0; i < TOTAL_NODE; i++){
        for(int j = 0; j < TOTAL_NODE; j++){
            if(i == j){
                cost[i][j] = 0.0;
                cost[j][i] = 0.0;
                phem[i][j] = 0.0;
                phem[j][i] = 0.0;
                continue;
            }
            if(j > i) {
                cost[i][j] = dis(generator);
                phem[i][j] = dis(generator);
                continue;
            }
            if(j < i) {
                cost[i][j] = cost[j][i];
                phem[i][j] = phem[j][i];
                continue;
            }
        }
    }


    // Print the cost matrix
    cout << "Cost Matrix:" << endl;
   //for(int i = 0; i < TOTAL_NODE; i++){
   //    for(int j = 0; j < TOTAL_NODE; j++){
   //        cout << cost[i][j] << " ";
   //    }
   //    cout << endl;
   //}


    //Print the phem matrix
    //cout << "Phem Matrix:" << endl;
    for(int i = 0; i < TOTAL_NODE; i++){
        for(int j = 0; j < TOTAL_NODE; j++){
            cout << phem[i][j] << " ";
        }
        cout << endl;
    }
}