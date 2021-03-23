#include <bits/stdc++.h>

using namespace std;

int n;
vector<pair<int, pair<int, int>>> edges;
vector<pair<int, int>> ans_edges;
vector<int> par, sz;

int find(int a){
    return (par[a] == a ? a : par[a] = find(par[a]));
}
void make(int a, int b){
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (sz[a] < sz[b]) swap(a, b);
    par[b] = a;
    sz[a] += sz[b];
}

int main () {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

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
    clock_t sttime = clock();
    sort(edges.begin(), edges.end());
    int ans = 0;
    for (auto &x : edges){
        int w = x.first;
        int a = x.second.first;
        int b = x.second.second;
        if (find(a) == find(b)) continue;
        ans_edges.push_back({a, b});
        ans += w;
        make(a, b);
    }
    sort(ans_edges.begin(), ans_edges.end());
    clock_t endtime = clock();
    cout << ans << endl;
    for (auto &x : ans_edges){
        cout << x.first << "-" << x.second << endl;
    }
    cout << "Waktu Eksekusi: " <<  (endtime - sttime) * 1000 / CLOCKS_PER_SEC << " ms" << endl;

    return 0;
}