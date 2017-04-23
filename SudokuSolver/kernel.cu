
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define BOARD_SIZE 9
#define SUB_BOARD_SIZE 3
#define MAX_NUM_BOARDS 32768 // 2 ^ 15 
#define NUM_ELEMENTS_PER_BOARD 81

///<summary>Returns an index used to access a 1D array, board[index3D(boardIdx, row, col)] == board[boardIdx][row][col]</summary>
///<param name="boardIdx">Index of board, index3D(4,0,0) will return an index of the first element of the 5th board</param>
///<param name="row">Row to be accessed at board at boardIdx</param>
///<param name="col">Column to be accessed at board at boardIdx</param>
__device__ __host__ inline int index3D(const int boardIdx, const int row, const int col)
{
	//Could replace NUM_ELEMENTS_PER_BOARD to BOARD_SIZE * BOARD_SIZE to have less preprocessor directives
	return boardIdx * NUM_ELEMENTS_PER_BOARD + (row * BOARD_SIZE) + col;
}

///<summary>Returns an index used to access a 1D array, board[index2D(row, col)] == board[row][col]</summary>
///<param name="row">Row to be accessed</param>
///<param name="col">Column to be accessed</param>
__device__ __host__ inline int index2D(const int row, const int col)
{
	return row * BOARD_SIZE + col;
}
///<summary>Prints the sudoku grid of BOARD_SIZE * BOARD_SIZE elements</summary>
///<param name="board">A pointer to the first integer of the sudoku board to be printed</param>
__device__ __host__ void printBoard(const int *board)
{

	int i;
	printf("Sudoku board: \n");
	for (i = 0; i < NUM_ELEMENTS_PER_BOARD; i++)
	{
		if (i % BOARD_SIZE == 0)
			printf("\n");
		printf("%d ", board[i]);
	}
	printf("\n");
}
///<summary>
///Resets an array of bools of length n to all false. <para />
///Used for checking the validity of a sudoku board 
///</summary>
///<param name="arr">A pointer to the first element of the bool array</param>
///<param name="n">Number of elements in the bool arr</param>
__device__ __host__ void resetBoolArr(bool *arr, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		arr[i] = false;
	}
}

///<summary>Checks if the board is a valid sudoku board (row constraint, column constraint, sub-board constraint)</summary>
///<param name="board">A pointer to the first integer of the sudoku board</param>
///<returns>True if board is valid, false otherwise</returns>
__device__  __host__ bool isBoardValid(const int *board)
{
	int i, j;
	const int BITMAP_SIZE = BOARD_SIZE + 1;
	bool seen[BITMAP_SIZE];
	resetBoolArr(seen, BITMAP_SIZE);
	//Rows are valid
	for (i = 0; i < BOARD_SIZE; i++)
	{
		resetBoolArr(seen, BITMAP_SIZE);
		for (j = 0; j < BOARD_SIZE; j++)
		{
			int val = board[index2D(i, j)];
			if (val != 0)
			{
				if (seen[val])
				{
					return false;
				}
				else
				{
					seen[val] = true;
				}
			}
		}
	}
	//Columns are valid
	for (i = 0; i < BOARD_SIZE; i++)
	{
		resetBoolArr(seen, BITMAP_SIZE);
		for (j = 0; j < BOARD_SIZE; j++)
		{
			int val = board[index2D(i, j)];
			if (val != 0)
			{
				if (seen[val])
				{
					return false;
				}
				else
				{
					seen[val] = true;
				}
			}
		}
	}
	//Sub-boards are valid
	for (int row = 0; row < SUB_BOARD_SIZE; row++)
	{
		for (int col = 0; col < SUB_BOARD_SIZE; col++)
		{
			resetBoolArr(seen, BITMAP_SIZE);
			for (i = 0; i < SUB_BOARD_SIZE; i++)
			{
				for (j = 0; j < SUB_BOARD_SIZE; j++)
				{
					int val = board[index2D(row * SUB_BOARD_SIZE + i, col * SUB_BOARD_SIZE + j)];
					if (val != 0)
					{
						if (seen[val])
						{
							return false;
						}

						else
						{
							seen[val] = true;
						}
					}
				}
			}
		}
	}

	return true;
}
///<summary>Checks if the board is a valid sudoku board, but is optimized such that it only checks row/col/sub-board of the changed element</summary>
///<param name="board">A pointer to the first integer of the sudoku board</param>
///<param name="changedRow">Index of the changed row</param>
///<param name="changedCol">Index of the changed column</param>
///<returns>True if board is valid, false otherwise</returns>
__device__ bool isBoardValid(const int *board, int changedRow, int changedCol)
{
	const int BITMAP_SIZE = BOARD_SIZE + 1;
	bool seen[BITMAP_SIZE];
	resetBoolArr(seen, BITMAP_SIZE);
	if (changedRow < 0 || changedCol < 0)
	{
		return isBoardValid(board); // nothing was changed
	}
	if (board[index2D(changedRow, changedCol)] < 1 || board[index2D(changedRow, changedCol)] > 9)
	{
		return false;
	}
	//Changed row is still valid
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		int val = board[index2D(changedRow, i)];
		if (val != 0)
		{
			if (seen[val])
			{
				return false;
			}
			else
			{
				seen[val] = true;
			}
		}
	}
	resetBoolArr(seen, BITMAP_SIZE);
	//Changed column is still valid
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		int val = board[index2D(i, changedCol)];
		if (val != 0)
		{
			if (seen[val])
			{
				return false;
			}
			else
			{
				seen[val] = true;
			}
		}
	}
	resetBoolArr(seen, BITMAP_SIZE);
	int r = changedRow / SUB_BOARD_SIZE;
	int c = changedCol / SUB_BOARD_SIZE;
	//Changed sub-board is still valid
	for (int i = 0; i < SUB_BOARD_SIZE; i++)
	{
		for (int j = 0; j < SUB_BOARD_SIZE; j++)
		{
			int val = board[index2D(r * SUB_BOARD_SIZE + i, c * SUB_BOARD_SIZE + j)];
			if (val != 0)
			{
				if (seen[val])
				{
					return false;
				}
				else
				{
					seen[val] = true;
				}
			}
		}

	}
	return true;
}
///<summary>
///Kernel function that finds new valid boards given a set of old boards. <para />
///Each threads works on its own old board and attempts to set some new values in the board's empty fields, <para />
///if the new value is valid it copies the whole newly found valid sudoku board to newBoards.<para />
///Once the board is copied we work on another old board at an index offset by the total number of threads in the program such that no 2 threads will work
///on the same board at the same time. <para /> 
///This function is essentially performing BFS (beadth-first search) because it searches the sudoku board "from left to right", ie
///it searches the first empty elements unlike DFS (depth-first search) which would check the last elements in the sudoku board first. <para />
///This function should be called by alternating the pointers "oldBoards" and "newBoards", 
///such that the newly found boards in one iteration will become the boards to be processed in the next iteration.
///</summary>
///<param name = "oldBoards">A pointer to the first element of the array of boards to be processed, size of array is MAX_NUM_OF_BOARDS * NUM_OF_ELEMENTS_PER_BOARD</param>
///<param name = "newBoards">A pointer to the first element of the array of newly found boards using BFS, size of array is MAX_NUM_OF_BOARDS * NUM_OF_ELEMENTS_PER_BOARD</param>
///<param name = "emptyFields">A pointer to the first element of the array that stores the 2D indices of empty fields of a given board,<para />
///size of array is MAX_NUM_OF_BOARDS * NUM_OF_ELEMENTS_PER_BOARD</param>
///<param name = "numOfOldBoards">Number of boards in oldBoards, used so we know when to finish the loop</param>
///<param name = "boardIndex">Number of boards in oldBoards, used so we know when to finish looping</param>
///<param name = "numOfEmptyFields">Number of empty fields at a given board index, numOfEmptyFields[3] == 10 means there is 10 empty fields in the 4th board</param>
__global__ void createPartialSolutionUsingBFS(
	int *oldBoards,
	int *newBoards,
	int *emptyFields,
	const int numOfOldBoards,
	int *boardIndex,
	int *numOfEmptyFields)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x; // represents the current sudoku board index
	//We have this condition so we do not overwrite the new set of valid boards

	while (tid < numOfOldBoards && tid < MAX_NUM_BOARDS)
	{
		bool foundNewBoard = false;
		// This loop starts at the first index of the board at index tid and loops through the whole board until its last element
		// For example, if tid == 2 and BOARD_SIZE == 9 then "i" will range from [162, 242] since index3D(2, 8, 8) = 2 * 81 + 8 * 9 + 8 = 242
		// Therefore the range [162, 242] represents all the indices of elements belonging to the third sudoku board
		// If we wish to pass a pointer to the first element of the board at index tid, we can do ptr + index3D(tid, 0, 0) which is 0 + tid * NUM_ELEMENTS_PER_BOARD + 0 * 0 + 0
		for (int i = index3D(tid, 0, 0); i <= index3D(tid, BOARD_SIZE - 1, BOARD_SIZE - 1) && !foundNewBoard; i++)
		{
			if (oldBoards[i] == 0)
			{
				foundNewBoard = true;
				for (int possibleValue = 1; possibleValue <= BOARD_SIZE; possibleValue++)
				{
					bool foundNewValidValue = false;
					oldBoards[i] = possibleValue;
					//We need to decode the row and column given a 3D index
					int temp = i - BOARD_SIZE * BOARD_SIZE * tid; //Substract the current index by the total amount of elements in all previous boards
					int r = temp / BOARD_SIZE;
					int c = temp % BOARD_SIZE;
					if (isBoardValid(oldBoards + index3D(tid, 0, 0), r, c))
					{
						foundNewValidValue = true;
					}
					else
					{
						oldBoards[i] = 0;
					}
					if (foundNewValidValue)
					{
						//We found a new valid sudoku value, so we copy the board to newBoards and find the indices of empty fields
						//emptyFields will help us in sudokuBacktrack() where we will skip to the first empty field in the board
						int nextBoardIndex = atomicAdd(boardIndex, 1);
						//printf("NBI: %d", nextBoardIndex);
						int currentEmptyIndex = 0;
						for (int row = 0; row < BOARD_SIZE; row++)
						{
							for (int col = 0; col < BOARD_SIZE; col++)
							{
								newBoards[index3D(nextBoardIndex, row, col)] = oldBoards[index3D(tid, row, col)];
								if (oldBoards[index3D(tid, row, col)] == 0)
								{
									emptyFields[index3D(nextBoardIndex, 0, currentEmptyIndex)] = index2D(row, col);
									currentEmptyIndex++;
								}
							}
						}
						numOfEmptyFields[nextBoardIndex] = currentEmptyIndex;
					}
				}
			}

		}
		tid += blockDim.x * gridDim.x; // offset by total number of threads in a given block
	}
}
///<summary>
///Kernel function that makes each thread in parallel run the sudoku backtracking algorithm, described here: https://en.wikipedia.org/wiki/Sudoku_solving_algorithms#Backtracking <para />
///When one thread finds the solution, it sets the "finished" flag to 1 and all other threads will be notified
///</summary>
///<param name = "boards">Pointer to the first integer of the array of boards to run backtracking on. Size is MAX_NUM_OF_ELEMENTS_PER_BOARD * NUM_OF_ELEMENTS_PER_BOARD</param>
///<param name = "numOfBoards">Number of boards of size NUM_OF_ELEMENTS_PER_BOARD in the "boards" array.</param>
///<param name = "numOfEmptyFields">Number of empty fields at a given board index, numOfEmptyFields[3] == 10 means there is 10 empty fields in the 4th board</param>
///<param name = "finished">Pointer to a single integer, if the value at of that int is 0 then board is not yet solved, 1 if it is solved.</param>
///<param name = "solvedBoard">Pointer to the first integer of the array of size NUM_OF_ELEMENTS_PER_BOARD, stores the solved sudoku board</param>
__global__ void sudokuBacktrack(
	int *boards,
	const int numOfBoards,
	int *emptyFields,
	int *numOfEmptyFields,
	int *finished,
	int *solvedBoard)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int numOfEmptyFieldsInThisBoard;

	while ((*finished == 0) && tid < numOfBoards && tid < MAX_NUM_BOARDS)
	{
		numOfEmptyFieldsInThisBoard = numOfEmptyFields[tid];

		int emptyIndex = 0;
		while (emptyIndex >= 0 && (emptyIndex < numOfEmptyFieldsInThisBoard))
		{
			int row = emptyFields[index3D(tid, 0, emptyIndex)] / BOARD_SIZE;
			int col = emptyFields[index3D(tid, 0, emptyIndex)] % BOARD_SIZE;
			// Increment the value of the empty field until it is valid
			boards[index3D(tid, row, col)]++;
			if (!isBoardValid(boards + index3D(tid, 0, 0), row, col))
			{
				if (boards[index3D(tid, row, col)] >= BOARD_SIZE)
				{
					//If we have tried all possible values we backtrack to the last empty field we changed and try a different value for it
					boards[index3D(tid, row, col)] = 0;
					emptyIndex--;
				}
			}
			else
			{
				// We have found a valid value for this field so we move forward in the backtracking algorithm
				emptyIndex++;
			}
			if (emptyIndex == numOfEmptyFieldsInThisBoard)
			{
				// We have filled all empty fields in the board with valid values so we have solved the board
				*finished = 1;
				printf("Thread at index %d has solved the board \n", tid);
				//Copy board to solvedBoard, which will later be copied back the host
				for (int r = 0; r < BOARD_SIZE; r++)
				{
					for (int c = 0; c < BOARD_SIZE; c++)
					{
						solvedBoard[index2D(r, c)] = boards[index3D(tid, r, c)];
					}
				}
			}

		}
		tid += gridDim.x * blockDim.x;
	}
}
///<summary>
///Initializes all data needed to call createPartialSolutionUsingBFS() and sudokuBacktrack()
///Then it runs the algorithm and prints the solved board
///</summary>
///<param name="numThreadsPerBlk">Number of threads in a block that will work</param>
///<param name="numBlocks">Number of total thread blocks that will work</param>
///<param name="inputBoard">Array of ints that stores the input board that we wish to solve</param>
///<returns>A value of type cudaError_t. cudaSuccess if no errors occured, cudaError otherwise</returns>
cudaError_t runParallelSudoku(
	const int numThreadsPerBlk,
	const int numBlocks,
	int *inputBoard)
{
	cudaError_t cudaStatus; // The return value of CUDA-library functions
	int bfsIterations = 20; // Number of times to run BFS to find some new valid boards
	int bfsBoardCount = 0; // The number of new boards we have found after a call to createPartialSolutionUsingBFS()
	int *boardIndex; // Must start at 0 every time
	// The meaning of the variables below has been described in the comments above createPartialSolutionUsingBFS() and sudokuBacktrack()
	int *numOfEmptyFields;
	int *finished = nullptr;
	int *newBoards;
	int *oldBoards;
	int *solvedBoard;
	int *dev_solvedBoard;
	int *emptyFields;
	int initialNumOfBoards = 1;
	solvedBoard = new int[NUM_ELEMENTS_PER_BOARD];
	memset(solvedBoard, 0, NUM_ELEMENTS_PER_BOARD);

	//Allocate memory for our boards
	cudaStatus = cudaMalloc(&newBoards, MAX_NUM_BOARDS * NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed for new boards! ");
		goto Error;
	}
	cudaStatus = cudaMalloc(&oldBoards, MAX_NUM_BOARDS * NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed for old boards! ");
		goto Error;
	}
	cudaStatus = cudaMalloc(&numOfEmptyFields, MAX_NUM_BOARDS * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed for numOfEmptyFields! ");
		goto Error;
	}
	cudaStatus = cudaMalloc(&emptyFields, MAX_NUM_BOARDS * NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed for numOfEmptyFields! ");
		goto Error;
	}
	cudaStatus = cudaMalloc(&boardIndex, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed for boardIndex! ");
		goto Error;
	}
	//Set memory to all zeros
	cudaStatus = cudaMemset(newBoards, 0, MAX_NUM_BOARDS * NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset failed for new boards! ");
		goto Error;
	}

	cudaStatus = cudaMemset(oldBoards, 0, MAX_NUM_BOARDS * NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset failed for old boards! ");
		goto Error;
	}

	cudaStatus = cudaMemset(boardIndex, 0, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset failed for boardIndex! ");
		goto Error;
	}
	cudaStatus = cudaMemset(emptyFields, 0, MAX_NUM_BOARDS * NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset failed for emptyFields! ");
		goto Error;
	}
	cudaStatus = cudaMemset(numOfEmptyFields, 0, MAX_NUM_BOARDS * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset failed for numOfEmptyFields! ");
		goto Error;
	}
	//Copy input board to oldBoards:
	cudaStatus = cudaMemcpy(oldBoards, inputBoard, NUM_ELEMENTS_PER_BOARD * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed for inputBoard -> oldBoards! ");
		goto Error;
	}

	createPartialSolutionUsingBFS << <numBlocks, numThreadsPerBlk >> >(oldBoards, newBoards, emptyFields, initialNumOfBoards, boardIndex, numOfEmptyFields);

	for (int i = 0; i < bfsIterations; i++)
	{
		cudaStatus = cudaMemcpy(&bfsBoardCount, boardIndex, sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed for boardIndex -> bfsBoardCount! on iteration %d", i);
			goto Error;
		}
		printf("Number of new boards found after iteration %d: %d\n", i, bfsBoardCount);

		cudaStatus = cudaMemset(boardIndex, 0, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemset failed for boardIndex! on iteration %d", i);
			goto Error;
		}

		if (i % 2 == 0)
		{
			createPartialSolutionUsingBFS << <numBlocks, numThreadsPerBlk >> >(newBoards, oldBoards, emptyFields, bfsBoardCount, boardIndex, numOfEmptyFields);
		}
		else
		{
			createPartialSolutionUsingBFS << <numBlocks, numThreadsPerBlk >> >(oldBoards, newBoards, emptyFields, bfsBoardCount, boardIndex, numOfEmptyFields);
		}
	}
	/////////////////////////////////////////////
	/////Done with BFS, now we run backtrack/////
	/////////////////////////////////////////////

	cudaStatus = cudaMemcpy(&bfsBoardCount, boardIndex, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for boardIndex -> bfsBoardCount! Before sudoku backtrack");
		goto Error;
	}
	cudaStatus = cudaMalloc(&finished, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for finished! ");
		goto Error;
	}
	cudaStatus = cudaMalloc(&dev_solvedBoard, NUM_ELEMENTS_PER_BOARD * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for dev_solvedBoard! ");
		goto Error;
	}
	cudaStatus = cudaMemset(finished, 0, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed for finished! ");
		goto Error;
	}
	if (bfsIterations % 2 == 1)
	{
		newBoards = oldBoards;
	}
	sudokuBacktrack << <numBlocks, numThreadsPerBlk >> >(newBoards, bfsBoardCount, emptyFields, numOfEmptyFields, finished, dev_solvedBoard);

	//Get solved board
	cudaStatus = cudaMemcpy(solvedBoard, dev_solvedBoard, NUM_ELEMENTS_PER_BOARD * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for dev_solvedBoard -> solvedBoard! ");
		goto Error;
	}

	printf("Solved board! \n");
	printBoard(solvedBoard);
	std::cout << "Is solved board valid? " << (isBoardValid(solvedBoard) ? "yes" : "no") << std::endl;
Error:
	cudaFree(finished);
	cudaFree(dev_solvedBoard);
	cudaFree(newBoards);
	cudaFree(oldBoards);
	cudaFree(emptyFields);
	cudaFree(boardIndex);
	cudaFree(numOfEmptyFields);
	free(solvedBoard);
	return cudaStatus;
}
int main(int argc, char** argv)
{
	// examples from http://www.websudoku.com
	int easyInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		0,6,0,3,0,0,8,0,4,
		5,3,7,0,9,0,0,0,0,
		0,4,0,0,0,6,3,0,7,
		0,9,0,0,5,1,2,3,8,
		0,0,0,0,0,0,0,0,0,
		7,1,3,6,2,0,0,4,0,
		0,0,0,0,6,0,5,2,3,
		1,0,2,0,0,9,0,8,0,
		3,0,6,0,0,2,0,0,0};
	int mediumInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		0,9,7,0,0,0,0,0,0,
		0,0,0,0,7,0,0,0,3,
		0,0,2,0,1,6,0,0,9,
		0,5,8,0,2,9,3,0,0,
		1,0,0,4,0,7,0,0,8,
		0,0,4,3,8,0,9,5,0,
		8,0,0,2,6,0,1,0,0,
		9,0,0,0,4,0,0,0,0,
		0,0,0,0,0,0,6,7,0};
	int hardInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		0,0,0,0,0,0,0,5,0,
		0,4,5,0,0,1,0,0,0,
		7,0,0,0,2,0,4,0,1,
		0,9,0,1,0,7,2,0,0,
		3,0,0,0,0,0,0,0,4,
		0,0,4,6,0,3,0,8,0,
		8,0,6,0,5,0,0,0,3,
		0,0,0,3,0,0,5,7,0,
		0,3,0,0,0,0,0,0,0};
	int veryHardInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		3,0,0,0,0,2,0,0,0,
		0,4,6,0,0,0,0,0,0,
		0,0,7,3,5,0,0,2,0,
		5,0,0,0,6,1,0,0,0,
		0,6,0,0,0,0,0,1,0,
		0,0,0,4,7,0,0,0,2,
		0,9,0,0,3,5,8,0,0,
		0,0,0,0,0,0,9,5,0,
		0,0,0,8,0,0,0,0,4};
	int allZeros[NUM_ELEMENTS_PER_BOARD] = {
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0 };
	if (argc != 3)
	{
		printf("Usage: argv[1] is threads per block, argv[2] is num of blocks\n");
		return 0;
	}
	const int threadsPerBlock = atoi(argv[1]);
	const int maxBlocks = atoi(argv[2]);

	printf("Threads per block: %d, num of blocks: %d \n", threadsPerBlock, maxBlocks);
	printf("Easy board: \n");
	printBoard(easyInputBoard);
	std::cout << "Is board valid? " << (isBoardValid(easyInputBoard) ? "yes" : "no") << std::endl;
	/////////////
	printf("Medium board: \n");
	printBoard(mediumInputBoard);
	std::cout << "Is medium board valid? " << (isBoardValid(mediumInputBoard) ? "yes" : "no") << std::endl;
	/////////////
	printf("Hard board: \n");
	printBoard(hardInputBoard);
	std::cout << "Is hard board valid? " << (isBoardValid(hardInputBoard) ? "yes" : "no") << std::endl;
	/////////////
	printf("Very hard board: \n");
	printBoard(veryHardInputBoard);
	std::cout << "Is very hard board valid? " << (isBoardValid(veryHardInputBoard) ? "yes" : "no") << std::endl;
	/////////////
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}		
	runParallelSudoku(threadsPerBlock, maxBlocks, easyInputBoard);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "bfs kernel launch failed for easy board: %s\n", cudaGetErrorString(cudaStatus));
	}	
	runParallelSudoku(threadsPerBlock, maxBlocks, mediumInputBoard);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
	fprintf(stderr, "bfs kernel launch failed for medium board: %s\n", cudaGetErrorString(cudaStatus));
	}
	runParallelSudoku(threadsPerBlock, maxBlocks, hardInputBoard);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "bfs kernel launch failed for hard board: %s\n", cudaGetErrorString(cudaStatus));
	}
	runParallelSudoku(threadsPerBlock, maxBlocks, veryHardInputBoard);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "bfs kernel launch failed for very hard: %s\n", cudaGetErrorString(cudaStatus));
	}
	runParallelSudoku(threadsPerBlock, maxBlocks, allZeros);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "bfs kernel launch failed for all zeros: %s\n", cudaGetErrorString(cudaStatus));
	}
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}


