#include <mpi.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

static void readMatrix(const string& filename, vector<vector<int>>& matrix, int& n) {
    ifstream file(filename);
    if (!file.is_open())
        throw invalid_argument("Couldn't open file " + filename);
    if (!(file >> n) || n <= 0)
        throw invalid_argument("Couldn't read the size from file: " + filename);
    matrix.assign(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(file >> matrix[i][j]))
                throw invalid_argument("Couldn't read an element from file " + filename);
}

static void writeMatrix(const string& filename, const vector<vector<int>>& C, int n) {
    ofstream f(filename);
    if (!f.is_open()) throw runtime_error("Couldn't create file " + filename);
    f << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f << C[i][j];
            if (j + 1 < n) f << " ";
        }
        f << "\n";
    }
}

// Последовательная эталонная проверка
static bool verifySequential(const vector<vector<int>>& A,
    const vector<vector<int>>& B,
    const vector<vector<int>>& C,
    int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            long long sum = 0;
            for (int k = 0; k < n; ++k)
                sum += (long long)A[i][k] * B[k][j];
            if (sum != (long long)C[i][j])
                return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Определение номера процесса и общего числа процессов
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Открытие файлов для записи (только rank 0, режим дописывания)
    ofstream timingFile, errorFile;
    if (rank == 0) {
        timingFile.open("timing.txt", ios::app);
        errorFile.open("errors.txt", ios::app);
    }

    // Запись ошибки в файл (rank 0) и аварийное завершение всех процессов
    auto logError = [&](const string& msg) {
        if (rank == 0 && errorFile.is_open()) {
            errorFile << "ERROR: " << msg << "\n";
            errorFile.flush();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        };

    try {
        // Проверка аргумента командной строки: размер матрицы n
        if (argc < 2)
            throw invalid_argument("Usage: multmatrix_mpi <n>");
        const int n = atoi(argv[1]);
        if (n <= 0)
            throw invalid_argument("Invalid size argument");

        string fileA = "matrix_a" + to_string(n) + ".txt";
        string fileB = "matrix_b" + to_string(n) + ".txt";
        string fileC = "matrix_c" + to_string(n) + ".txt";

        vector<vector<int>> A, B;
        if (rank == 0) {
            int nA = 0, nB = 0;
            readMatrix(fileA, A, nA);
            readMatrix(fileB, B, nB);
            if (nA != n || nB != n)
                throw invalid_argument("File size does not match requested n");
        }

        // Подготовка матрицы B к рассылке всем процессам
        vector<int> Bflat((size_t)n * n);
        if (rank == 0)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Bflat[(size_t)i * n + j] = B[i][j];

        // Разбиение строк матрицы A
        vector<int> counts(size), displs(size);
        int base = n / size, rem = n % size;
        for (int r = 0, off = 0; r < size; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }

        // Плоский буфер A на rank 0 для Scatterv
        vector<int> Aflat;
        if (rank == 0) {
            Aflat.resize((size_t)n * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Aflat[(size_t)i * n + j] = A[i][j];
        }

        // Пересчёт counts/displs из строк в элементы для Scatterv
        vector<int> cntD(size), dspD(size);
        for (int r = 0; r < size; ++r) {
            cntD[r] = counts[r] * n;
            dspD[r] = displs[r] * n;
        }

        vector<int> localA((size_t)counts[rank] * n);

        // Синхронизация перед замером: все процессы готовы
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        // Рассылка B всем процессам
        MPI_Bcast(Bflat.data(), (int)Bflat.size(), MPI_INT, 0, MPI_COMM_WORLD);

        // Раздача полос строк A: каждый процесс получает свои counts[rank] строк
        MPI_Scatterv(Aflat.data(), cntD.data(), dspD.data(), MPI_INT,
            localA.data(), cntD[rank], MPI_INT,
            0, MPI_COMM_WORLD);

        vector<int> localC((size_t)counts[rank] * n, 0);

        for (int i = 0; i < counts[rank]; ++i) {
            for (int k = 0; k < n; ++k) {
                int aik = localA[(size_t)i * n + k];
                if (aik == 0) continue;
                for (int j = 0; j < n; ++j)
                    localC[(size_t)i * n + j] += aik * Bflat[(size_t)k * n + j];
            }
        }

        // Пересчёт counts/displs из строк в элементы для Gatherv
        vector<int> cntG(size), dspG(size);
        for (int r = 0; r < size; ++r) {
            cntG[r] = counts[r] * n;
            dspG[r] = displs[r] * n;
        }

        // Буфер для сбора результата на rank 0
        vector<int> Cflat;
        if (rank == 0) Cflat.resize((size_t)n * n, 0);

        // Сбор полос C со всех процессов на rank 0
        MPI_Gatherv(localC.data(), (int)localC.size(), MPI_INT,
            Cflat.data(), cntG.data(), dspG.data(), MPI_INT,
            0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        // Финиш: локальное время, затем max по всем процессам
        double t_end = MPI_Wtime();
        double local_elapsed = t_end - t_start;

        double elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Разворачивание плоского Cflat в двумерную матрицу C на rank 0
        if (rank == 0) {
            vector<vector<int>> C(n, vector<int>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = Cflat[(size_t)i * n + j];

            writeMatrix(fileC, C, n);

            // Проверка корректности (только для n <= 800)
            const int VERIFY_MAX_N = 800;
            string checkStr;
            double verifyTime = 0.0;

            if (n <= VERIFY_MAX_N) {
                auto vstart = high_resolution_clock::now();
                bool ok = verifySequential(A, B, C, n);
                auto vend = high_resolution_clock::now();
                verifyTime = duration<double>(vend - vstart).count();
                checkStr = ok ? "OK" : "FAIL";
            }
            else {
                checkStr = "SKIP";
            }

            // Запись результата в timing.txt (для SKIP - без verify_time)
            if (timingFile.is_open()) {
                timingFile << "P=" << size
                    << " | n=" << n
                    << " | time=" << elapsed << " sec"
                    << " | check=" << checkStr;
                if (checkStr != "SKIP")
                    timingFile << " | verify_time=" << verifyTime << " sec";
                timingFile << "\n";
                timingFile.flush();
            }
        }
    }
    catch (const exception& e) {
        logError(e.what());
    }

    // Закрытие файлов на rank 0 и завершение MPI
    if (rank == 0) {
        if (timingFile.is_open()) timingFile.close();
        if (errorFile.is_open()) errorFile.close();
    }

    MPI_Finalize();
    return 0;
}