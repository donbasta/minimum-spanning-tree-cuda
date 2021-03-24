%%writefile MST_CUDA.cu

#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <string>

using namespace std;

struct Edge {
    int w, a, b;
    Edge(int w, int a, int b) : w(w), a(a), b(b) {}
};

int n;
vector<Edge> edges;
vector<int> par, sz;

int find(int a) {
    return (par[a] == a ? a : par[a] = find(par[a]));
}
void make(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (sz[a] < sz[b]) swap(a, b);
    par[b] = a;
    sz[a] += sz[b];
}

__device__ int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

__device__ void merge_edges_by_weight(Edge* ar, Edge* temp, int le, int mid, int ri) {
    int cur = le;
    int i = le;
    int j = mid;
    while (i < mid && j < ri) {
        bool compare = 0;
        if (ar[i].w != ar[j].w) compare = (ar[i].w < ar[j].w);
        else if (ar[i].a != ar[i].a) compare = (ar[i].a < ar[j].a);
        else compare = (ar[i].b < ar[j].b);

        if (compare)
            temp[cur++] = ar[i++];
        else
            temp[cur++] = ar[j++];
    }
    while (i < mid) {
        temp[cur++] = ar[i++];
    }
    while (j < ri) {
        temp[cur++] = ar[j++];
    }
}

__device__ void merge_edges_lexicographically(Edge* ar, Edge* temp, int le, int mid, int ri) {
    int cur = le;
    int i = le;
    int j = mid;
    while (i < mid && j < ri) {
        bool compare = 0;
        if (ar[i].a != ar[j].a) compare = (ar[i].a < ar[j].a);
        else compare = (ar[i].b < ar[j].b);
        if (compare)
            temp[cur++] = ar[i++];
        else
            temp[cur++] = ar[j++];
    }
    while (i < mid) {
        temp[cur++] = ar[i++];
    }
    while (j < ri) {
        temp[cur++] = ar[j++];
    }
}

__global__ void sort_edges_by_weight(Edge *ar, Edge *temp, int len, int gap, int slices, dim3* threads, dim3* blocks) {
    int index = getIdx(threads, blocks);
    int le = gap * index * slices;

    for (int slice = 0; slice < slices; slice++) {
        if (le >= len) continue;

        int mid = min(le + gap / 2, len);
        int ri = min(le + gap, len);

        merge_edges_by_weight(ar, temp, le, mid, ri);

        le += gap;
    }
}

__global__ void sort_edges_lexicographically(Edge *ar, Edge *temp, int len, int gap, int slices, dim3* threads, dim3* blocks) {
    int index = getIdx(threads, blocks);
    int le = gap * index * slices;

    for (int slice = 0; slice < slices; slice++) {
        if (le >= len) continue;

        int mid = min(le + gap / 2, len);
        int ri = min(le + gap, len);

        merge_edges_lexicographically(ar, temp, le, mid, ri);

        le += gap;
    }
}

int main(int argc, char** argv) {

    cin >> n;
    par.resize(n);
    for (int i = 0;i < n;i++) {
        par[i] = i;
    }
    sz.resize(n, 1);
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            int w;
            cin >> w;
            if (w >= 0 && i < j) edges.emplace_back(w, i, j);
        }
    }

    int blockSize = 32;
    if (argc > 1) {
        blockSize = stoi(argv[1]);
    }

    cudaSetDevice(0);
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 1;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = blockSize;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    dim3* D_threads;
    dim3* D_blocks;

    cudaMallocManaged((void**) &D_threads, sizeof(dim3));
    cudaMallocManaged((void**) &D_blocks, sizeof(dim3));

    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    int len = edges.size();
    Edge *ar, *temp, *data;
    cudaMalloc(&ar, len * sizeof(Edge));
    cudaMalloc(&temp, len * sizeof(Edge));
    data = (Edge*) malloc(len * sizeof(Edge));
    for (int i = 0; i < len; i++) {
        data[i] = edges[i];
    }
    cudaMemcpy(ar, data, len * sizeof(Edge), cudaMemcpyHostToDevice);

    int numThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                      blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    Edge *ar_1 = ar;
    Edge *temp_1 = temp;

    clock_t sttime = clock();

    cudaEvent_t start_sort_1, end_sort_1;
    cudaEventCreate(&start_sort_1);
    cudaEventCreate(&end_sort_1);

    cudaEventRecord(start_sort_1);
    for (int gap = 2;gap < len * 2;gap <<= 1) {
        int slices = len / ((numThreads) * gap) + 1;
        sort_edges_by_weight<<<blocksPerGrid, threadsPerBlock>>>(ar_1, temp_1, len, gap, slices, D_threads, D_blocks);
        cudaDeviceSynchronize();
        ar_1 = ar_1 == ar ? temp : ar;
        temp_1 = temp_1 == ar ? temp : ar;
    }
    cudaEventRecord(end_sort_1);
    cudaEventSynchronize(end_sort_1);
    float time_sort_1 = 0;
    cudaEventElapsedTime(&time_sort_1, start_sort_1, end_sort_1);


    cudaMemcpy(data, ar_1, len * sizeof(Edge), cudaMemcpyDeviceToHost);
    
    int ans = 0;
    Edge* ans_edges;
    ans_edges = (Edge*) malloc(len * sizeof(Edge));
    {
        int id = 0;
        for (int i = 0;i < len;i++) {
            if (find(data[i].a) == find(data[i].b)) continue;
            ans_edges[id++] = data[i];
            ans += data[i].w;
            make(data[i].a, data[i].b);
        }
        len = id;
    }
    
    cudaMemcpy(ar, ans_edges, len * sizeof(Edge), cudaMemcpyHostToDevice);

    cudaEvent_t start_sort_2, end_sort_2;
    cudaEventCreate(&start_sort_2);
    cudaEventCreate(&end_sort_2);

    ar_1 = ar;
    temp_1 = temp;

    cudaEventRecord(start_sort_2);

    for (int gap = 2;gap < len * 2;gap <<= 1) {
        int slices = len / ((numThreads) * gap) + 1;
        sort_edges_lexicographically<<<blocksPerGrid, threadsPerBlock>>>(ar_1, temp_1, len, gap, slices, D_threads, D_blocks);
        cudaDeviceSynchronize();
        ar_1 = ar_1 == ar ? temp : ar;
        temp_1 = temp_1 == ar ? temp : ar;
    }
    cudaEventRecord(end_sort_2);
    cudaEventSynchronize(end_sort_2);
    float time_sort_2 = 0;
    cudaEventElapsedTime(&time_sort_2, start_sort_2, end_sort_2);

    cudaMemcpy(ans_edges, ar_1, len * sizeof(Edge), cudaMemcpyDeviceToHost);

    clock_t endtime = clock();
    cout << ans << endl;
    for (int i = 0;i < len;i++) {
        cout << ans_edges[i].a << "-" << ans_edges[i].b << endl;
    }
    cout << "Waktu Eksekusi: " << (endtime - sttime) * 1000 / CLOCKS_PER_SEC << " ms" << endl;

    cout << "Waktu Eksekusi sorting 1 " << time_sort_1 << " ms" << endl;
    cout << "Waktu Eksekusi sorting 2 " << time_sort_2 << " ms" << endl;

    cudaFree(ar);
    cudaFree(temp);
    cudaFree(D_threads);
    cudaFree(D_blocks);
    free(ans_edges);
    free(data);

    return 0;
}
