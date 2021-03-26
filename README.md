 # Checkers AI
 
 ## Introduction

 It uses an 8x8 checkered gameboard. In this attempt to create a game agent, a tree traversal approach has been used. Checkers is a 1vs1 zero-sum game. Minimax algorithm is best suited for such types of games. ⍺-β pruning is used to improve performance.

 ## Evaluation functions

All our evaluation functions can be divided into 2 parts – the main in-game part (opening-middlegame) and the
ending part. In the first part we try to reach some optimal stage (not necessarily the
end of the game). 

### Opening-Middlegame evaluation functions
Specifically:

We split the board into halves.

Pawn in the opponent's half of the board value = 7

Pawn in the player's half of the board value = 5

King’s value = 10 

### Ending eval functions

For each piece (king) of the player we sum all the distances between it and all the opponent’s pieces. If the
player has more kings than the opponent he will prefer a game position that minimizes this sum (he wants to
attack), otherwise he will prefer this sum to be as big as possible (run away). 