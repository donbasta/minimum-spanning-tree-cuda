#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

struct Edge{
    int w, a, b;
};

bool comparator_by_weight(const Edge& a, const Edge& b){
    if (a.w != b.w) return a.w < b.w;
    if (a.a != b.a) return a.a < b.a;
    return a.b < b.b;
}
bool comparator_lexicographic(const Edge& a, const Edge& b){
    if (a.a != b.a) return a.a < b.a;
    return a.b < b.b;
}

int n;
vector<pair<int, pair<int, int>>> edges;
vector<int> par, sz;

int main(){
    cin >> n;
    par.resize(n);
    for (int i=0;i<n;i++){
        par[i] = i;
    }
    sz.resize(n, 1);
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            int w;
            cin >> w;
            if (w > 0 && i < j) edges.push_back({w, {i, j}});
        }
    }
    int len = edges.size();
    Edge *ar; 
    cudaMallocManaged(&ar, len * sizeof(Edge));
    // TODO

    return 0;
}