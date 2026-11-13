#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void readMatrix(const string& filename, vector<vector<int>>& matrix, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw invalid_argument("Couldn't open file: " + filename);
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
    const int NUM_THREADS = 8;
    try {
        omp_set_num_threads(NUM_THREADS);
        vector<vector<int>> A, B;
        int nA, nB;

        readMatrix("matrix_a.txt", A, nA);
        readMatrix("matrix_b.txt", B, nB);

        if (nA != nB) {
            throw invalid_argument("The sizes of the matrices do not match");
        }

        int n = nA;

        ofstream results("results.txt", ios::app);
        if (!results.is_open()) {
            cerr << "Couldn't open results.txt" << endl;
            return 1;
        }
        results << fixed << setprecision(4);

        double bestTime = 1e18;

        for (int run = 0; run < 5; run++) {
            vector<vector<int>> C(n, vector<int>(n, 0));

            auto start = high_resolution_clock::now();

            #pragma omp parallel for
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

            if (elapsed.count() < bestTime) {
                bestTime = elapsed.count();
            }

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
        }

        results << "Size: " << n << "x" << n
            << " | Threads: " << NUM_THREADS
            << " | Time: " << bestTime << " sec\n";
        results.flush();

        results.close();
    }
    catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    return 0;
}