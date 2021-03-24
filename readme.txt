File readme.txt yang berisi penjelasan mengenai program yang dibuat termasuk logika atau ide dari paralelisasinya. 

Program MST paralel yang dibuat menggunakan algoritma Kruskal.
Paralelisasi dilakukan pada operasi sorting, dan algoritma sorting yang digunakan adalah merge sort.
Sorting dilakukan sebanyak dua kali, yakni saat melakukan sorting seluruh edge pada graf (by weight), dan sorting seluruh edge pada MST secara leksikografis untuk dicetak.

Apabila N merupakan jumlah node, maka sorting pertama melibatkan O(N^2) elemen dan sorting kedua hanya melibatkan N-1 elemen saja, sehingga waktu eksekusi sorting pertama signifikan terhadap total runtime program (bottleneck).

Berikut ini merupakan kinerja algoritma yang dijalankan pada Google Colab.
Spesifikasi Test Case: 3000 Node 

Sebagai benchmark, waktu eksekusi algoritma MST serial pada kasus uji yang digunakan adalah  

Hasil eksekusi:

| Jumlah Node  | Merge Sort 1 (ms) | Merge Sort 2 (ms) | Total Runtime (ms) |
| ------------- | ------------- |
| 1  | 19064.10 | 5.69 | 19204 |
| 2  | 11097.40 | 3.34 | 11237 |
| 4  | 6593.01 | 2.15 | 6732 |
| 8  | 4458.3 | 1.62 | 4605 |
| 16  | 3440.55 | 1.38 | 3583 |
| 32  | 2946.07 | 1.27 | 3084 |
| 64  | 2704.41 | 1.23 | 2844 |
| 128  | 2625.18 | 1.21 | 2765 |
| 256 | 2550.18 | 1.19 | 2692 | 
