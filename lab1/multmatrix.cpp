#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

void readMatrix(const string& filename, vector<vector<int>>& matrix, int& n) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw invalid_argument("Couldn't open file " + filename);
	}
	if (!(file >> n) || n <= 0) {
		throw invalid_argument("Couldn't read the size from file: " + filename);
	}
	matrix.assign(n, vector<int>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (!(file >> matrix[i][j])) {
				throw invalid_argument("Couldn't read an element from file");
			}
		}
	}
}

int main() {
	try {
		vector<vector<int>> A, B;
		int nA, nB;

		readMatrix("matrix_a.txt", A, nA);
		readMatrix("matrix_b.txt", B, nB);

		if (nA != nB) {
			throw invalid_argument("The sizes of the matrices do not match");
		}

		int n = nA;

		for (int run = 0; run < 5; run++) {
			vector<vector<int>> C(n, vector<int>(n, 0));

			auto start = high_resolution_clock::now();

			for (int i = 0; i < n; ++i) {
				for (int k = 0; k < n; ++k) {
					int aik = A[i][k];
					for (int j = 0; j < n; ++j) {
						C[i][j] += aik * B[k][j];
					}
				}
			}

			auto end = high_resolution_clock::now();
			duration<double> elapsed = end - start;

			ofstream fileC("matrix_c.txt");
			if (!fileC.is_open()) {
				cerr << "Couldn't create file matrix_c.txt" << endl;
				return 1;
			}
			fileC << n << "\n";
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {
					fileC << C[i][j];
					if (j + 1 < n) fileC << " ";
				}
				fileC << "\n";
			}
			fileC.close();

			cout << "Size: " << n << "x" << n << " | Time: " << elapsed.count() << " sec\n";
		}
	}
	catch (const exception& e) {
		cerr << "ERROR: " << e.what() << endl;
		return 1;
	}
	return 0;
}