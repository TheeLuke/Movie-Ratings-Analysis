### Final Results

- LSH Parameters Used: Num Permutations=256, Jaccard Threshold=0.3
- Min Common Ratings for Correlation: 15
- Filtering: Users with < 20 ratings, Movies with < 15 ratings (adjust values in find_movie_twins_minhash)

#### Number of 'Movie Twin' pairs found by MinHash & Jaccard (up to top 100): 100
- Average Pearson Correlation for these 'Movie Twin' Pairs: 0.3123
- Average Pearson Correlation for Randomly Selected Pairs: 0.2214

```
--- Top 100 'Movie Twin' Pairs (UserID1, UserID2, Jaccard Similarity) ---
1. User IDs: (18356, 313668), Jaccard Similarity: 1.0000
2. User IDs: (169845, 308490), Jaccard Similarity: 1.0000
3. User IDs: (97685, 227884), Jaccard Similarity: 1.0000
4. User IDs: (169493, 265259), Jaccard Similarity: 1.0000
5. User IDs: (18356, 265259), Jaccard Similarity: 1.0000
6. User IDs: (93572, 177183), Jaccard Similarity: 1.0000
7. User IDs: (236540, 266093), Jaccard Similarity: 1.0000
8. User IDs: (77318, 170112), Jaccard Similarity: 1.0000
9. User IDs: (92534, 234310), Jaccard Similarity: 1.0000
10. User IDs: (177183, 313463), Jaccard Similarity: 1.0000
11. User IDs: (234310, 313668), Jaccard Similarity: 1.0000
12. User IDs: (13806, 199013), Jaccard Similarity: 1.0000
13. User IDs: (47942, 77318), Jaccard Similarity: 1.0000
14. User IDs: (177837, 249077), Jaccard Similarity: 1.0000
15. User IDs: (124266, 224936), Jaccard Similarity: 1.0000
16. User IDs: (18356, 92534), Jaccard Similarity: 1.0000
17. User IDs: (86856, 181279), Jaccard Similarity: 1.0000
18. User IDs: (141201, 266093), Jaccard Similarity: 1.0000
19. User IDs: (125829, 243711), Jaccard Similarity: 1.0000
20. User IDs: (124266, 291625), Jaccard Similarity: 1.0000
21. User IDs: (224936, 291625), Jaccard Similarity: 1.0000
22. User IDs: (265259, 313668), Jaccard Similarity: 1.0000
23. User IDs: (12125, 170112), Jaccard Similarity: 1.0000
24. User IDs: (47942, 248242), Jaccard Similarity: 1.0000
25. User IDs: (187276, 195935), Jaccard Similarity: 1.0000
26. User IDs: (158283, 240989), Jaccard Similarity: 1.0000
27. User IDs: (147789, 279965), Jaccard Similarity: 1.0000
28. User IDs: (3799, 170112), Jaccard Similarity: 1.0000
29. User IDs: (236540, 277862), Jaccard Similarity: 1.0000
30. User IDs: (47942, 65538), Jaccard Similarity: 1.0000
31. User IDs: (169493, 234310), Jaccard Similarity: 1.0000
32. User IDs: (136065, 322263), Jaccard Similarity: 1.0000
33. User IDs: (18356, 234310), Jaccard Similarity: 1.0000
34. User IDs: (38929, 322547), Jaccard Similarity: 1.0000
35. User IDs: (12125, 65538), Jaccard Similarity: 1.0000
36. User IDs: (141201, 277862), Jaccard Similarity: 1.0000
37. User IDs: (92534, 169493), Jaccard Similarity: 1.0000
38. User IDs: (3799, 65538), Jaccard Similarity: 1.0000
39. User IDs: (65538, 248242), Jaccard Similarity: 1.0000
40. User IDs: (12125, 47942), Jaccard Similarity: 1.0000
41. User IDs: (266093, 277862), Jaccard Similarity: 1.0000
42. User IDs: (195935, 293880), Jaccard Similarity: 1.0000
43. User IDs: (170112, 248242), Jaccard Similarity: 1.0000
44. User IDs: (77318, 248242), Jaccard Similarity: 1.0000
45. User IDs: (22437, 50132), Jaccard Similarity: 1.0000
46. User IDs: (12125, 77318), Jaccard Similarity: 1.0000
47. User IDs: (38643, 173449), Jaccard Similarity: 1.0000
48. User IDs: (3799, 77318), Jaccard Similarity: 1.0000
49. User IDs: (234310, 265259), Jaccard Similarity: 1.0000
50. User IDs: (169493, 313668), Jaccard Similarity: 1.0000
51. User IDs: (187276, 293880), Jaccard Similarity: 1.0000
52. User IDs: (65538, 170112), Jaccard Similarity: 1.0000
53. User IDs: (165007, 224200), Jaccard Similarity: 1.0000
54. User IDs: (296007, 299674), Jaccard Similarity: 1.0000
55. User IDs: (12125, 248242), Jaccard Similarity: 1.0000
56. User IDs: (92534, 313668), Jaccard Similarity: 1.0000
57. User IDs: (18356, 169493), Jaccard Similarity: 1.0000
58. User IDs: (56180, 264494), Jaccard Similarity: 1.0000
59. User IDs: (3799, 248242), Jaccard Similarity: 1.0000
60. User IDs: (92534, 265259), Jaccard Similarity: 1.0000
61. User IDs: (141201, 236540), Jaccard Similarity: 1.0000
62. User IDs: (47942, 170112), Jaccard Similarity: 1.0000
63. User IDs: (93572, 313463), Jaccard Similarity: 1.0000
64. User IDs: (3799, 12125), Jaccard Similarity: 1.0000
65. User IDs: (60252, 329440), Jaccard Similarity: 1.0000
66. User IDs: (3799, 47942), Jaccard Similarity: 1.0000
67. User IDs: (119572, 123530), Jaccard Similarity: 1.0000
68. User IDs: (65538, 77318), Jaccard Similarity: 1.0000
69. User IDs: (77647, 294432), Jaccard Similarity: 0.9984
70. User IDs: (241273, 296245), Jaccard Similarity: 0.9910
71. User IDs: (25084, 86967), Jaccard Similarity: 0.9866
72. User IDs: (86967, 294432), Jaccard Similarity: 0.9835
73. User IDs: (77647, 86967), Jaccard Similarity: 0.9820
74. User IDs: (25084, 294432), Jaccard Similarity: 0.9819
75. User IDs: (167541, 173768), Jaccard Similarity: 0.9818
76. User IDs: (25084, 77647), Jaccard Similarity: 0.9804
77. User IDs: (50012, 174815), Jaccard Similarity: 0.9770
78. User IDs: (41012, 300048), Jaccard Similarity: 0.9761
79. User IDs: (159785, 319480), Jaccard Similarity: 0.9750
80. User IDs: (48673, 107031), Jaccard Similarity: 0.9750
81. User IDs: (53426, 330736), Jaccard Similarity: 0.9714
82. User IDs: (242889, 271504), Jaccard Similarity: 0.9709
83. User IDs: (115400, 144661), Jaccard Similarity: 0.9677
84. User IDs: (195935, 319748), Jaccard Similarity: 0.9667
85. User IDs: (71302, 147789), Jaccard Similarity: 0.9667
86. User IDs: (71302, 279965), Jaccard Similarity: 0.9667
87. User IDs: (187276, 319748), Jaccard Similarity: 0.9667
88. User IDs: (293880, 319748), Jaccard Similarity: 0.9667
89. User IDs: (13012, 56180), Jaccard Similarity: 0.9667
90. User IDs: (13012, 264494), Jaccard Similarity: 0.9667
91. User IDs: (71302, 138579), Jaccard Similarity: 0.9655
92. User IDs: (32448, 259106), Jaccard Similarity: 0.9643
93. User IDs: (245387, 249077), Jaccard Similarity: 0.9600
94. User IDs: (178961, 200620), Jaccard Similarity: 0.9600
95. User IDs: (132494, 266741), Jaccard Similarity: 0.9600
96. User IDs: (177837, 245387), Jaccard Similarity: 0.9600
97. User IDs: (13195, 124266), Jaccard Similarity: 0.9600
98. User IDs: (13195, 224936), Jaccard Similarity: 0.9600
99. User IDs: (125637, 204124), Jaccard Similarity: 0.9600
100. User IDs: (13195, 291625), Jaccard Similarity: 0.9600
```