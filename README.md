# Simple-AlphaZero
Simple AlphaZero implementation on solved toy problem, demonstrating strengths quickly (&lt;1hr runtime on decent CPU)

# Performance
The model can beat the alpha-beta solver running at depth 10, playing both first and second. It also plays nearly optimally according to the website connect4.gamesolver.org. 

# Training
The training took just over 30 minutes on my CPU (as the GPU is only used to update the weights here). 

# Logs
```
$ python alphazero_connect4.py
1. Train AlphaZero
2. Play Human vs AlphaZero
3. Play AlphaZero vs Alpha-Beta Solver
4. Play AlphaZero vs Perfect/Solved Bot
Enter choice (1, 2, 3, or 4): 1
Training on: cuda | CPU Workers: 19

--- Iteration 1/100 ---
Self-play generated 4133 moves in 17.4s
Loss: 2.9901
Evaluating current model vs Random...
Eval vs Random | Wins: 18 | Losses: 2 | Draws: 0 | Win Rate: 90.0%

--- Iteration 2/100 ---
Self-play generated 4286 moves in 17.9s
Loss: 2.9275
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 3/100 ---
Self-play generated 4252 moves in 17.9s
Loss: 2.8084
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 4/100 ---
Self-play generated 3832 moves in 15.7s
Loss: 2.6676
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 5/100 ---
Self-play generated 4339 moves in 18.2s
Loss: 2.5542
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 6/100 ---
Self-play generated 4561 moves in 18.5s
Loss: 2.4136
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 7/100 ---
Self-play generated 4027 moves in 16.2s
Loss: 2.3047
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 8/100 ---
Self-play generated 3884 moves in 15.5s
Loss: 2.2404
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 9/100 ---
Self-play generated 4653 moves in 18.1s
Loss: 2.1789
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 10/100 ---
Self-play generated 4446 moves in 17.1s
Loss: 2.1155
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 11/100 ---
Self-play generated 4745 moves in 18.8s
Loss: 2.0473
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 12/100 ---
Self-play generated 4965 moves in 19.7s
Loss: 2.0023
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 13/100 ---
Self-play generated 4771 moves in 19.7s
Loss: 1.9530
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 14/100 ---
Self-play generated 4803 moves in 18.6s
Loss: 1.9085
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 15/100 ---
Self-play generated 4532 moves in 18.4s
Loss: 1.8868
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 16/100 ---
Self-play generated 4619 moves in 19.1s
Loss: 1.8562
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 17/100 ---
Self-play generated 5085 moves in 21.3s
Loss: 1.8247
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 18/100 ---
Self-play generated 4994 moves in 19.4s
Loss: 1.8010
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 19/100 ---
Self-play generated 5044 moves in 19.9s
Loss: 1.7608
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 20/100 ---
Self-play generated 4947 moves in 19.3s
Loss: 1.7231
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 21/100 ---
Self-play generated 4971 moves in 20.8s
Loss: 1.7044
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 22/100 ---
Self-play generated 4764 moves in 19.2s
Loss: 1.6790
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 23/100 ---
Self-play generated 4531 moves in 17.2s
Loss: 1.6682
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 24/100 ---
Self-play generated 4619 moves in 19.1s
Loss: 1.6559
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 25/100 ---
Self-play generated 4791 moves in 19.9s
Loss: 1.6293
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 26/100 ---
Self-play generated 4742 moves in 17.4s
Loss: 1.6337
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 27/100 ---
Self-play generated 5008 moves in 19.4s
Loss: 1.6359
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 28/100 ---
Self-play generated 5488 moves in 21.5s
Loss: 1.6052
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 29/100 ---
Self-play generated 5456 moves in 20.5s
Loss: 1.5861
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 30/100 ---
Self-play generated 5332 moves in 20.5s
Loss: 1.5542
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 31/100 ---
Self-play generated 5218 moves in 20.2s
Loss: 1.5517
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 32/100 ---
Self-play generated 5646 moves in 22.3s
Loss: 1.5337
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 33/100 ---
Self-play generated 5605 moves in 21.8s
Loss: 1.4991
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 34/100 ---
Self-play generated 5773 moves in 21.7s
Loss: 1.4679
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 35/100 ---
Self-play generated 5376 moves in 20.5s
Loss: 1.4531
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 36/100 ---
Self-play generated 5185 moves in 20.5s
Loss: 1.4408
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 37/100 ---
Self-play generated 5121 moves in 20.9s
Loss: 1.4343
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 38/100 ---
Self-play generated 5446 moves in 20.5s
Loss: 1.4126
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 39/100 ---
Self-play generated 5205 moves in 20.0s
Loss: 1.3917
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 40/100 ---
Self-play generated 5296 moves in 20.5s
Loss: 1.4067
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 41/100 ---
Self-play generated 5243 moves in 20.0s
Loss: 1.3976
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 42/100 ---
Self-play generated 5146 moves in 20.5s
Loss: 1.3874
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 43/100 ---
Self-play generated 5250 moves in 20.0s
Loss: 1.3562
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 44/100 ---
Self-play generated 5459 moves in 21.1s
Loss: 1.3476
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 45/100 ---
Self-play generated 5505 moves in 21.5s
Loss: 1.3461
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 46/100 ---
Self-play generated 5488 moves in 21.1s
Loss: 1.3585
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 47/100 ---
Self-play generated 5449 moves in 21.3s
Loss: 1.3516
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 48/100 ---
Self-play generated 5725 moves in 23.0s
Loss: 1.3434
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 49/100 ---
Self-play generated 5751 moves in 22.1s
Loss: 1.3356
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 50/100 ---
Self-play generated 5791 moves in 22.7s
Loss: 1.3075
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 51/100 ---
Self-play generated 5952 moves in 21.5s
Loss: 1.3042
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 52/100 ---
Self-play generated 5639 moves in 20.5s
Loss: 1.2955
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 53/100 ---
Self-play generated 5911 moves in 21.5s
Loss: 1.2854
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 54/100 ---
Self-play generated 6020 moves in 22.7s
Loss: 1.2482
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 55/100 ---
Self-play generated 5458 moves in 19.8s
Loss: 1.2286
Evaluating current model vs Random...
Eval vs Random | Wins: 19 | Losses: 1 | Draws: 0 | Win Rate: 95.0%

--- Iteration 56/100 ---
Self-play generated 5548 moves in 21.4s
Loss: 1.2274
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 57/100 ---
Self-play generated 5849 moves in 21.3s
Loss: 1.2226
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 58/100 ---
Self-play generated 5369 moves in 20.9s
Loss: 1.2144
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 59/100 ---
Self-play generated 5480 moves in 20.6s
Loss: 1.1980
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 60/100 ---
Self-play generated 5586 moves in 22.1s
Loss: 1.1929
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 61/100 ---
Self-play generated 5089 moves in 20.5s
Loss: 1.1865
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 62/100 ---
Self-play generated 5660 moves in 21.5s
Loss: 1.1858
Evaluating current model vs Random...
Eval vs Random | Wins: 19 | Losses: 1 | Draws: 0 | Win Rate: 95.0%

--- Iteration 63/100 ---
Self-play generated 5305 moves in 20.5s
Loss: 1.1714
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 64/100 ---
Self-play generated 5570 moves in 21.2s
Loss: 1.1749
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 65/100 ---
Self-play generated 5392 moves in 20.2s
Loss: 1.1821
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 66/100 ---
Self-play generated 5333 moves in 20.2s
Loss: 1.1887
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 67/100 ---
Self-play generated 5234 moves in 19.8s
Loss: 1.1754
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 68/100 ---
Self-play generated 5431 moves in 21.3s
Loss: 1.1772
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 69/100 ---
Self-play generated 5454 moves in 20.2s
Loss: 1.1864
Evaluating current model vs Random...
Eval vs Random | Wins: 19 | Losses: 1 | Draws: 0 | Win Rate: 95.0%

--- Iteration 70/100 ---
Self-play generated 5429 moves in 20.7s
Loss: 1.2273
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 71/100 ---
Self-play generated 5695 moves in 21.6s
Loss: 1.2215
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 72/100 ---
Self-play generated 5599 moves in 20.9s
Loss: 1.2233
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 73/100 ---
Self-play generated 5917 moves in 22.4s
Loss: 1.2105
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 74/100 ---
Self-play generated 5392 moves in 21.0s
Loss: 1.1957
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 75/100 ---
Self-play generated 5752 moves in 23.1s
Loss: 1.2037
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 76/100 ---
Self-play generated 5524 moves in 21.1s
Loss: 1.2129
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 77/100 ---
Self-play generated 5525 moves in 20.5s
Loss: 1.2005
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 78/100 ---
Self-play generated 5178 moves in 20.1s
Loss: 1.1840
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 79/100 ---
Self-play generated 5323 moves in 20.2s
Loss: 1.1794
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 80/100 ---
Self-play generated 5438 moves in 21.2s
Loss: 1.1866
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 81/100 ---
Self-play generated 5418 moves in 21.2s
Loss: 1.1993
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 82/100 ---
Self-play generated 5335 moves in 21.2s
Loss: 1.2068
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 83/100 ---
Self-play generated 5185 moves in 19.5s
Loss: 1.2145
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 84/100 ---
Self-play generated 5089 moves in 19.2s
Loss: 1.2161
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 85/100 ---
Self-play generated 5266 moves in 19.6s
Loss: 1.2230
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 86/100 ---
Self-play generated 5521 moves in 20.0s
Loss: 1.2322
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 87/100 ---
Self-play generated 5202 moves in 19.8s
Loss: 1.2291
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 88/100 ---
Self-play generated 5325 moves in 21.1s
Loss: 1.2220
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 89/100 ---
Self-play generated 5305 moves in 20.9s
Loss: 1.2146
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 90/100 ---
Self-play generated 5190 moves in 28.4s
Loss: 1.2198
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 91/100 ---
Self-play generated 5352 moves in 30.2s
Loss: 1.2250
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 92/100 ---
Self-play generated 5200 moves in 20.3s
Loss: 1.2182
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 93/100 ---
Self-play generated 5356 moves in 19.9s
Loss: 1.2219
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 94/100 ---
Self-play generated 5626 moves in 22.0s
Loss: 1.2174
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 95/100 ---
Self-play generated 5522 moves in 21.0s
Loss: 1.2157
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 96/100 ---
Self-play generated 5501 moves in 21.1s
Loss: 1.2140
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 97/100 ---
Self-play generated 5564 moves in 21.0s
Loss: 1.1909
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 98/100 ---
Self-play generated 5291 moves in 19.8s
Loss: 1.1704
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 99/100 ---
Self-play generated 5830 moves in 21.7s
Loss: 1.1725
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%

--- Iteration 100/100 ---
Self-play generated 5909 moves in 21.6s
Loss: 1.1818
Evaluating current model vs Random...
Eval vs Random | Wins: 20 | Losses: 0 | Draws: 0 | Win Rate: 100.0%
Training complete.
```
