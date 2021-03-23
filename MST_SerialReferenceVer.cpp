#include <bits/stdc++.h>

using namespace std;

struct Edge{
    int w, a, b;
    Edge(int w, int a, int b) : w(w), a(a), b(b) {}
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
vector<Edge> edges;
vector<Edge> ans_edges;
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
            if (w >= 0 && i < j) edges.emplace_back(w, i, j);
        }
    }
    clock_t sttime = clock();
    int len = edges.size();
    // Iterative Merge Sort
    for (int gap=2;gap<len * 2;gap<<=1){
        for (int st=0;st<len;st+=gap){
            int i = st;
            int mid = i + gap / 2;
            if (mid >= len) continue;
            int j = mid;
            int ri = min(mid + gap / 2, len);

            vector<Edge> temp;
            while (i < mid && j < ri){
                if (comparator_by_weight(edges[i], edges[j])) 
                    temp.emplace_back(edges[i++]);
                else
                    temp.emplace_back(edges[j++]);
            }
            while (i < mid){
                temp.emplace_back(edges[i++]);
            }
            while (j < ri){
                temp.emplace_back(edges[j++]);
            }
            for (int k=st;k<ri;k++){
                edges[k] = temp[k - st];
            }
        }
    }
    int ans = 0;
    for (auto &x : edges){
        if (find(x.a) == find(x.b)) continue;
        ans_edges.emplace_back(x);
        ans += x.w;
        make(x.a, x.b);
    }
    len = ans_edges.size();
    // Iterative Merge Sort
    for (int gap=2;gap<len * 2;gap<<=1){
        for (int st=0;st<len;st+=gap){
            int i = st;
            int mid = i + gap / 2;
            if (mid >= len) continue;
            int j = mid;
            int ri = min(mid + gap / 2, len);

            vector<Edge> temp;
            while (i < mid && j < ri){
                if (comparator_lexicographic(ans_edges[i], ans_edges[j])) 
                    temp.emplace_back(ans_edges[i++]);
                else
                    temp.emplace_back(ans_edges[j++]);
            }
            while (i < mid){
                temp.emplace_back(ans_edges[i++]);
            }
            while (j < ri){
                temp.emplace_back(ans_edges[j++]);
            }
            for (int k=st;k<ri;k++){
                ans_edges[k] = temp[k - st];
            }
        }
    }
    clock_t endtime = clock();
    cout << ans << endl;
    for (auto &x : ans_edges){
        cout << x.a << "-" << x.b << endl;
    }
    cout << "Waktu Eksekusi: " <<  (endtime - sttime) * 1000 / CLOCKS_PER_SEC << " ms" << endl;

    return 0;
}