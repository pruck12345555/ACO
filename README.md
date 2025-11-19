1. Input cost adjacency matrix in the cost.txt file
2. Input pheromone adjacency matrix in the pheromone.txt file
3. If wanting to make the program prints out final pheromone matrix,
    3.1 Uncomment line 226 on acoParallel.cu
    3.2 Uncomment line 94 on acoSequential.cu
   otherwise, comment the lines then recompile  
4. Compile by nvcc acoSequential.cu -o acoSequential for sequential and nvcc acoParallel.cu -o acoParallel for parallel
5. Run by ./acoSequential.exe or ./acoParallel.exe