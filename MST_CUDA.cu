#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <time.h>

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

__global__ void sort_edges_by_weight(Edge *ar, Edge *temp, int len, int gap) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int le = index * gap;le < len;le += stride * gap) {
        int i = le;
        int mid = i + gap / 2;
        if (mid >= len) continue;
        int j = mid;
        int ri = min(mid + gap / 2, len);

        int cur = le;
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
        for (int k = le;k < ri;k++) {
            ar[k] = temp[k];
        }
    }
}
__global__ void sort_edges_lexicographically(Edge* ar, Edge* temp, int len, int gap) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int le = index * gap;le < len;le += stride * gap) {
        int i = le;
        int mid = i + gap / 2;
        if (mid >= len) continue;
        int j = mid;
        int ri = min(mid + gap / 2, len);

        int cur = le;
        while (i < mid && j < ri) {
            bool compare = 0;
            if (ar[i].a != ar[i].a) compare = (ar[i].a < ar[j].a);
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
        for (int k = le;k < ri;k++) {
            ar[k] = temp[k];
        }
    }
}

int main() {
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
    cudaSetDevice(0);
    clock_t sttime = clock();
    int len = edges.size();
    Edge *ar, *temp;
    cudaMallocManaged(&ar, len * sizeof(Edge));
    cudaMallocManaged(&temp, len * sizeof(Edge));

    for (int i = 0;i < len;i++) {
        ar[i] = edges[i];
    }

    int blockSize = 32;
    for (int gap = 2;gap < len * 2;gap <<= 1) {
        sort_edges_by_weight<<<1, blockSize>>>(ar, temp, len, gap);
        cudaDeviceSynchronize();
    }

    int ans = 0;
    Edge* ans_edges;
    cudaMallocManaged(&ans_edges, len * sizeof(Edge));
    {
        int id = 0;
        for (int i = 0;i < len;i++) {
            if (find(ar[i].a) == find(ar[i].b)) continue;
            ans_edges[id++] = ar[i];
            ans += ar[i].w;
            make(ar[i].a, ar[i].b);
        }
        len = id;
    }
    for (int gap = 2;gap < len * 2;gap <<= 1) {
        sort_edges_lexicographically<<<1, blockSize>>>(ans_edges, temp, len, gap);
        cudaDeviceSynchronize();
    }
    clock_t endtime = clock();
    cout << ans << endl;
    for (int i = 0;i < len;i++) {
        cout << ans_edges[i].a << "-" << ans_edges[i].b << endl;
    }
    cout << "Waktu Eksekusi: " << (endtime - sttime) * 1000 / CLOCKS_PER_SEC << " ms" << endl;

    cudaFree(ar);
    cudaFree(temp);
    cudaFree(ans_edges);

    return 0;
}