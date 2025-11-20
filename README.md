6610406057 ศุภวัชร สงแก้ว  
1. Edit total nodes, total ants by editing values of TOTAL_NODE and TOTAL_ANT at the top of each file
2. Input cost adjacency matrix in the cost.txt file (default 100 nodes)
3. Input pheromone adjacency matrix in the pheromone.txt file (default 100 nodes)
4. To make the program prints out final pheromone matrix then recompile  
    4.1 Uncomment line 231 on acoParallel.cu  
    4.2 Uncomment line 94 on acoSequential.cu  
5. Compile by nvcc acoSequential.cu -o acoSequential for sequential and nvcc acoParallel.cu -o acoParallel for parallel
6. Run by ./acoSequential.exe and ./acoParallel.exe  
  6.1 To redirect output into a textFile, do ./aco\<Parallel,Sequential\>.exe> > \<textFile>