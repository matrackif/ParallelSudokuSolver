#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>

#define BOARD_SIZE 9
#define SUB_BOARD_SIZE 3
#define NUM_ELEMENTS_PER_BOARD 81
using namespace std::chrono;
using namespace std;
inline int index2D(const int row, const int col)
{
	return row * BOARD_SIZE + col;
}
void printBoard(const int *board)
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
void resetBoolArr(bool *arr, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		arr[i] = false;
	}
}
bool isBoardValid(const int *board)
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
int firstEmptyIndex(int *board)
{
	for (int i = 0; i < NUM_ELEMENTS_PER_BOARD; i++)
	{
		if (board[i] == 0)
			return i;
	}

	return -1;
}
bool recursiveBacktrack(int *board)
{
	if (isBoardValid(board))
	{
		int index = firstEmptyIndex(board);
		if (index == -1)
			return true;
		for (int i = 1; i <= BOARD_SIZE; i++)
		{
			board[index] = i;
			if (recursiveBacktrack(board))
			{
				return true; // solved
			}
			board[index] = 0;
		}
	}
	return false;
}
int main(int argc, char** argv)
{
	int easyInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		0,6,0,3,0,0,8,0,4,
		5,3,7,0,9,0,0,0,0,
		0,4,0,0,0,6,3,0,7,
		0,9,0,0,5,1,2,3,8,
		0,0,0,0,0,0,0,0,0,
		7,1,3,6,2,0,0,4,0,
		0,0,0,0,6,0,5,2,3,
		1,0,2,0,0,9,0,8,0,
		3,0,6,0,0,2,0,0,0 };
	int mediumInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		0,9,7,0,0,0,0,0,0,
		0,0,0,0,7,0,0,0,3,
		0,0,2,0,1,6,0,0,9,
		0,5,8,0,2,9,3,0,0,
		1,0,0,4,0,7,0,0,8,
		0,0,4,3,8,0,9,5,0,
		8,0,0,2,6,0,1,0,0,
		9,0,0,0,4,0,0,0,0,
		0,0,0,0,0,0,6,7,0 };
	int hardInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		0,0,0,0,0,0,0,5,0,
		0,4,5,0,0,1,0,0,0,
		7,0,0,0,2,0,4,0,1,
		0,9,0,1,0,7,2,0,0,
		3,0,0,0,0,0,0,0,4,
		0,0,4,6,0,3,0,8,0,
		8,0,6,0,5,0,0,0,3,
		0,0,0,3,0,0,5,7,0,
		0,3,0,0,0,0,0,0,0 };
	int veryHardInputBoard[NUM_ELEMENTS_PER_BOARD] = {
		3,0,0,0,0,2,0,0,0,
		0,4,6,0,0,0,0,0,0,
		0,0,7,3,5,0,0,2,0,
		5,0,0,0,6,1,0,0,0,
		0,6,0,0,0,0,0,1,0,
		0,0,0,4,7,0,0,0,2,
		0,9,0,0,3,5,8,0,0,
		0,0,0,0,0,0,9,5,0,
		0,0,0,8,0,0,0,0,4 };
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
	int hardForBruteForce[NUM_ELEMENTS_PER_BOARD] = {
		0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,3,0,8,5,
		0,0,1,0,2,0,0,0,0,
		0,0,0,5,0,7,0,0,0,
		0,0,4,0,0,0,1,0,0,
		0,9,0,0,0,0,0,0,0,
		5,0,0,0,0,0,0,7,3,
		0,0,2,0,1,0,0,0,0,
		0,0,0,0,4,0,0,0,9
	};
	bool couldSolveBoard = false;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	couldSolveBoard = recursiveBacktrack(easyInputBoard);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	if (couldSolveBoard)
	{
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Easy board solved in " << duration << "us" << endl;
		printBoard(easyInputBoard);
	}
	t1 = high_resolution_clock::now();
	couldSolveBoard = recursiveBacktrack(mediumInputBoard);
	t2 = high_resolution_clock::now();
	if (couldSolveBoard)
	{
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Medium board solved in " << duration << "us" << endl;
		printBoard(mediumInputBoard);
	}
	t1 = high_resolution_clock::now();
	couldSolveBoard = recursiveBacktrack(hardInputBoard);
	t2 = high_resolution_clock::now();
	if (couldSolveBoard)
	{
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Hard board solved in " << duration << "us" << endl;
		printBoard(hardInputBoard);
	}
	t1 = high_resolution_clock::now();
	couldSolveBoard = recursiveBacktrack(veryHardInputBoard);
	t2 = high_resolution_clock::now();
	if (couldSolveBoard)
	{
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Very hard for human board solved in " << duration << "us" << endl;
		printBoard(veryHardInputBoard);
	}
	t1 = high_resolution_clock::now();
	couldSolveBoard = recursiveBacktrack(allZeros);
	t2 = high_resolution_clock::now();
	if (couldSolveBoard)
	{
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Empty board solved in " << duration << "us" << endl;
		printBoard(allZeros);
	}

	t1 = high_resolution_clock::now();
	couldSolveBoard = recursiveBacktrack(hardForBruteForce);
	t2 = high_resolution_clock::now();
	if (couldSolveBoard)
	{
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Hard for brute force board solved in " << duration << "us" << endl;
		printBoard(hardForBruteForce);
	}
	
	return 0;
}