#pragma once
#include <cstdint>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <chrono>
#include <memory>
#include <stdio.h>
#include <iomanip>
#include <cmath> // for M_E
#include <cstdlib>  // std::system

#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_cblas.h"

// Function to clear sparse matrix
void clearSparseMatrix(sparse_matrix_t& matrix);
void clearAndReleaseMemory(std::vector<float>& vec);
void clearAndReleaseMKL_INT(std::vector<MKL_INT>& vec);
double saveEmb(const std::string& filename, sparse_matrix_t csrA, double runtime,int timestep,int R, int K);
void saveX(const std::string& filename, sparse_matrix_t csrA, int R, int K);
int readValue(const std::string& filename, std::vector<float>& values);
int readIndicesOrIndptr(const std::string& filename, std::vector<MKL_INT>& col_indx);
void readMatrixInfo(const std::string& filename, MKL_INT& rows, MKL_INT& cols, MKL_INT& nnz);

int main(int argc, char* argv[]){
    std::string dataset;
    int R = 1;
    int K = 200;
    int start_year = 0;

    for (int i = 1; i < argc; ++i) { // Start with 1, because argv[0] is the program name
        std::string arg = argv[i];
        if (arg == "--dataset") {
            if (i + 1 < argc) { // Make sure you have the next parameter as the value
                dataset = argv[++i]; // Gets the value and adds the index
            } else {
                std::cerr << "--dataset option requires one argument." << std::endl;
                return 1;
            }
        } else if (arg == "--R") {
            if (i + 1 < argc) {
                R = std::atoi(argv[++i]);
            } else {
                std::cerr << "--R option requires one argument." << std::endl;
                return 1;
            }
        }else if (arg == "--K") {
            if (i + 1 < argc) {
                K = std::atoi(argv[++i]);
            } else {
                std::cerr << "--K option requires one argument." << std::endl;
                return 1;
            }
        }else if (arg == "--start_year") {
            if (i + 1 < argc) {
                start_year = std::atoi(argv[++i]);
            } else {
                std::cerr << "--start_year option requires one argument." << std::endl;
                return 1;
            }
        }
    }

    std::string data_path = "data/" + dataset;
    // Output the obtained parameters and verify that they are correct
    std::cout << "Dataset: " << dataset << std::endl;
    std::cout << "R: " << R << std::endl;

    float alpha = 1.0; // Scalar multiple factor of addition
    std::chrono::duration<double> duration;
    sparse_status_t status;
    double seconds, finalRuntime;

    MKL_INT nodes_new, attrs_new;
    MKL_INT rowsG, colsG, nnzG, rowsA, colsA, nnzA, rowsP, colsP, nnzP;
    std::vector<float> valuesG, valuesA, valuesP;
    std::vector<MKL_INT> indicesG, indicesA, indicesP;
    std::vector<MKL_INT> indptrG, indptrA, indptrP;

    sparse_matrix_t G, A, P, H;

    std::cout << "The time step is ： " << start_year << std::endl;
    // Read the data and create the adjacency matrix
    readMatrixInfo(data_path +"/network/info_" + std::to_string(start_year) + ".txt", rowsG, colsG, nnzG);
    std::cout << "nG:"<<rowsG<<" mG:"<<colsG<<" nnzG:"<<nnzG<< std::endl;
    readValue(data_path + "/network/values_" + std::to_string(start_year) + ".txt", valuesG);
    readIndicesOrIndptr(data_path +"/network/indices_" + std::to_string(start_year) + ".txt", indicesG);
    readIndicesOrIndptr(data_path +"/network/indptr_" + std::to_string(start_year) + ".txt", indptrG);
    status = mkl_sparse_s_create_csr(&G, SPARSE_INDEX_BASE_ZERO, rowsG, colsG, indptrG.data(), indptrG.data() + 1, indicesG.data(), valuesG.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSR matrix.\n");
        return 1;
    }
    printf("Succeeded in creating G:\n");

    //     Read the data and create the attributes matrix
    readMatrixInfo(data_path +"/attrs/info_" + std::to_string(start_year) + ".txt", rowsA, colsA, nnzA);
    std::cout << "nA:"<<rowsA<<" mA:"<<colsA<<" nnzA:"<<nnzA<< std::endl;
    readValue(data_path +"/attrs/values_" + std::to_string(start_year) + ".txt",valuesA);
    readIndicesOrIndptr(data_path +"/attrs/indices_" + std::to_string(start_year) + ".txt",indicesA);
    readIndicesOrIndptr(data_path +"/attrs/indptr_" + std::to_string(start_year) + ".txt",indptrA);
    status = mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, rowsA, colsA, indptrA.data(), indptrA.data() + 1, indicesA.data(), valuesA.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSR matrix.\n");
        return 1;
    }
    printf("Succeeded in creating A:\n");

    // Read the data and create the Sparse Random matrix
    readMatrixInfo(data_path +"/SRMatrix/info_" + std::to_string(start_year) + "_K" + std::to_string(K) + ".txt", rowsP, colsP, nnzP);
    readValue(data_path +"/SRMatrix/values_" + std::to_string(start_year) + "_K" + std::to_string(K) + ".txt",valuesP);
    readIndicesOrIndptr(data_path +"/SRMatrix/indices_" + std::to_string(start_year) + "_K" + std::to_string(K) + ".txt",indicesP);
    readIndicesOrIndptr(data_path +"/SRMatrix/indptr_" + std::to_string(start_year) + "_K" + std::to_string(K) + ".txt",indptrP);
    status = mkl_sparse_s_create_csr(&P, SPARSE_INDEX_BASE_ZERO, rowsP, colsP, indptrP.data(), indptrP.data() + 1, indicesP.data(), valuesP.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSR matrix.\n");
        return 1;
    }
    printf("Succeeded in creating P:\n");

    printf("Data reading ends and sparse random projection begins...\n");
    // Record the start time
    auto start = std::chrono::high_resolution_clock::now();
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, P, &H);
    // Record end time point
    auto end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    saveX(dataset, H, 0, K);

    clearSparseMatrix(P);
    clearSparseMatrix(A);
    clearAndReleaseMemory(valuesA);
    clearAndReleaseMKL_INT(indicesA);
    clearAndReleaseMKL_INT(indptrA);
    clearAndReleaseMemory(valuesP);
    clearAndReleaseMKL_INT(indicesP);
    clearAndReleaseMKL_INT(indptrP);

    for(int i = 1; i < R + 1; i++) {
        start = std::chrono::high_resolution_clock::now();
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, G, H, &H);
        end = std::chrono::high_resolution_clock::now();
        duration = duration + end - start;
        saveX(dataset, H, i, K);
    }

    seconds = duration.count();
    std::cout << "embedding time of VLS2Sketch " << seconds << " seconds." << std::endl;
    std::string emb_path = "results/" + dataset;
    std::cout << "Save embedding...... " << std::endl;
    finalRuntime = saveEmb(emb_path, H, seconds, start_year, R, K);
    std::cout << "final runtime of VLS2Sketch " << finalRuntime << " seconds." << std::endl;

    clearSparseMatrix(H);
    clearSparseMatrix(G);
    clearAndReleaseMemory(valuesG);
    clearAndReleaseMKL_INT(indicesG);
    clearAndReleaseMKL_INT(indptrG);
    return 0;
}

// Function to clear sparse matrix
void clearSparseMatrix(sparse_matrix_t& matrix) {
    if (matrix != nullptr) {
        mkl_sparse_destroy(matrix);
        matrix = nullptr;
    }
    mkl_free_buffers();
}
void clearAndReleaseMemory(std::vector<float>& vec) {
    // Clear the vector's contents
    vec.clear();
    // Request to shrink to fit (may not always release memory)
    vec.shrink_to_fit();
    // Optionally swap with an empty vector to forcefully release memory
    std::vector<float>().swap(vec);
}
void clearAndReleaseMKL_INT(std::vector<MKL_INT>& vec) {
    // Clear the vector's contents
    vec.clear();
    // Request to shrink to fit (may not always release memory)
    vec.shrink_to_fit();
    // Optionally swap with an empty vector to forcefully release memory
    std::vector<MKL_INT>().swap(vec);
}
double saveEmb(const std::string& filename, sparse_matrix_t csrA, double runtime,int timestep,int R, int K) {
    sparse_index_base_t indexing;
    MKL_INT nrows;
    MKL_INT ncols;
    MKL_INT* col_indptr_start;
    MKL_INT* col_indptr_end;
    MKL_INT* indices;
    float* csr_values;
    std::vector<bool> values;
    mkl_sparse_s_export_csr(csrA, &indexing, &nrows, &ncols, &col_indptr_start, &col_indptr_end, &indices, &csr_values);

    std::string emb_path = filename + "/emb_cpp";
    std::string time_path = filename + "/runtimes";
    std::string command = "mkdir -p " + emb_path;
    if (std::system(command.c_str()) == 0) {
        std::cout << "Successfully created a folder!" << std::endl;
    } else {
        std::cout << "An error occurred or the folder already exists!" << std::endl;
    }
    std::string command2 = "mkdir -p " + time_path;
    if (std::system(command2.c_str()) == 0) {
        std::cout << "Successfully created a folder!" << std::endl;
    } else {
        std::cout << "An error occurred or the folder already exists!" << std::endl;
    }

    std::string indices_path = emb_path + "/indices_R" + std::to_string(R) + "_timestep" + std::to_string(timestep) + "_K" + std::to_string(K) + ".txt";
    std::string indptr_path = emb_path + "/indptr_R" + std::to_string(R) + "_timestep" + std::to_string(timestep) + "_K" + std::to_string(K) + ".txt";
    std::string values_path = emb_path + "/values_R" + std::to_string(R) + "_timestep" + std::to_string(timestep) + "_K" + std::to_string(K) + ".txt";
    std::string info_path = emb_path + "/info_R" + std::to_string(R) + "_timestep" + std::to_string(timestep) + "_K" + std::to_string(K) + ".txt";
    std::string runtime_path = time_path + "/runtime_R" + std::to_string(R) + "_timestep" + std::to_string(timestep) + "_K" + std::to_string(K) + ".txt";

    // Create a file flow object
    std::ofstream indicesFile(indices_path);
    std::ofstream indptrFile(indptr_path);
    std::ofstream valuesFile(values_path);
    std::ofstream infoFile(info_path);
    std::ofstream runtimeFile(runtime_path);

    // Record the time when quantization began
    auto start = std::chrono::high_resolution_clock::now();
     // Record the end time point of quantization
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the time difference and convert it to seconds
    std::chrono::duration<double> duration = end - start;


    std::vector<MKL_INT> new_indices;
    std::vector<bool> new_values;
    std::vector<MKL_INT> new_indptr(nrows + 1, 0);

    for (MKL_INT i = 0; i < nrows; ++i) {
        for (MKL_INT j = col_indptr_start[i]; j < col_indptr_start[i + 1]; ++j) {
            start = std::chrono::high_resolution_clock::now();
            bool value = csr_values[j] > 0;
            end = std::chrono::high_resolution_clock::now();
            duration += end - start;
            if (value) {
                new_indices.push_back(indices[j]);
                new_values.push_back(true);
                ++new_indptr[i + 1];
            }
        }
        new_indptr[i + 1] += new_indptr[i];
    }

    double seconds = duration.count();
    seconds = seconds + runtime;
    std::cout << "final runtime of VLS2ketch " << seconds << " seconds." << std::endl;
    // Write the final time to the runtime.txt file
    runtimeFile << seconds << " (s)" << std::endl;


    for (MKL_INT i = 0; i <= nrows; i++) {
        indptrFile << new_indptr[i] << (i < nrows ? " " : "");
    }
    indptrFile << std::endl;

    for (MKL_INT i = 0; i < new_indices.size(); ++i) {
        indicesFile << new_indices[i] << (i < new_indices.size() - 1 ? " " : "");
        valuesFile << new_values[i] << (i < new_values.size() - 1 ? " " : "");
    }
    indicesFile << std::endl;
    valuesFile << std::endl;

    long long int actual_nnz = new_indices.size();

    // Save the number of rows, columns, and non-zero elements to the info.txt file
    infoFile << nrows << " " << ncols << " " << actual_nnz << std::endl;

    // Close the files
    indicesFile.close();
    indptrFile.close();
    valuesFile.close();
    infoFile.close();
    return seconds;
}
void saveX(const std::string& filename, sparse_matrix_t csrA, int R, int K) {
    sparse_index_base_t indexing;
    MKL_INT nrows;
    MKL_INT ncols;
    MKL_INT* col_indptr_start;
    MKL_INT* col_indptr_end;
    MKL_INT* indices;
    float* csr_values;
    mkl_sparse_s_export_csr(csrA, &indexing, &nrows, &ncols, &col_indptr_start, &col_indptr_end, &indices, &csr_values);

    std::string folder_path = "X_list/" + filename;

    std::string command = "mkdir -p " + folder_path;
    if (std::system(command.c_str()) == 0) {
        std::cout << "Successfully created a folder!" << std::endl;
    } else {
        std::cout << "An error occurred or the folder already exists!" << std::endl;
    }

    std::string indices_path = folder_path + "/indices_R" + std::to_string(R) + "_K" + std::to_string(K) + ".txt";
    std::string indptr_path = folder_path + "/indptr_R" + std::to_string(R) + "_K" + std::to_string(K) + ".txt";
    std::string values_path = folder_path + "/values_R" + std::to_string(R) + "_K" + std::to_string(K) + ".txt";
    std::string info_path = folder_path + "/info_R" + std::to_string(R) + "_K" + std::to_string(K) + ".txt";

    // Create a file flow object
    std::ofstream indicesFile(indices_path);
    std::ofstream indptrFile(indptr_path);
    std::ofstream valuesFile(values_path);
    std::ofstream infoFile(info_path);

    for (int i = 0; i <= nrows; i++) {
        indptrFile << col_indptr_start[i] << (i < nrows ? " " : "");
    }
    indptrFile << std::endl;

    long long int actual_nnz = col_indptr_start[nrows]; // The value of the nrows position represents the actual number of non-zero elements

    for (int i = 0; i < actual_nnz; i++) {
        indicesFile << indices[i] << (i < actual_nnz - 1 ? " " : "");
        valuesFile << csr_values[i] << (i < actual_nnz - 1 ? " " : "");
    }
    indicesFile << std::endl;
    valuesFile << std::endl;

    // Save the number of rows, columns, and non-zero elements to the info.txt file
    infoFile << nrows << " " << ncols << " " << actual_nnz << std::endl;

    // Close the files
    indicesFile.close();
    indptrFile.close();
    valuesFile.close();
    infoFile.close();
}
int readValue(const std::string& filename, std::vector<float>& values) {
    std::ifstream infile(filename);
    float val;
    if (!infile.is_open()) {
        std::cerr << "Unable to open file for reading." << std::endl;
        return 1;
    }
    values.clear();
    while (infile >> val) {
        values.push_back(val);
    }
    infile.close();
    return 0;
}
int readIndicesOrIndptr(const std::string& filename, std::vector<MKL_INT>& col_indx) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to open file for reading." << std::endl;
        return 1;
    }
    col_indx.clear(); // Make sure the vector is empty
    MKL_INT val;
    while (infile >> val) {
        col_indx.push_back(val);
    }
    infile.close();
    return 0;
}
void readMatrixInfo(const std::string& filename, MKL_INT& rows, MKL_INT& cols, MKL_INT& nnz) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    // Read the number of rows, columns, and non-zero elements directly
    if (!(file >> rows >> cols >> nnz)) {
        std::cerr << "Failed to read matrix info from file: " << filename << std::endl;
        return;
    }
    file.close();
}