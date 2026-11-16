# Music Recommendations with Collaborative Filtering & a Neural Network from Scratch

A two-part machine learning project:

- **Part 1: Recommender system.** Recommend artists to radio listeners using **item-item**, **user-user**, and **hybrid** collaborative filtering based on cosine similarity.
- **Part 2: Neural network in NumPy.** Build a 2-layer neural network from scratch (forward propagation, cross-entropy cost, backpropagation, gradient descent) and study how the learning rate affects weight updates.

---

## Repository Contents

| File | Description |
|---|---|
| `Part 1.jpynb` | Part 1: collaborative filtering recommender |
| `Part 2.ipynb` | Part 2: neural network built with NumPy |
| `radio_songs.csv` | User × artist listening data |

---

## Part 1: Collaborative Filtering Recommender

### Dataset
`radio_songs.csv` is a binary user × artist matrix:
- **100 users × 284 artists** (1 = the user has listened to the artist, 0 = not)
- No missing values

### A. Item-Item CF: listeners of U2 and Pink Floyd
1. Transpose the data so each artist is a row (284 × 100).
2. Compute a 284 × 284 **artist-artist cosine similarity matrix**.
3. Average each artist's similarity to U2 and to Pink Floyd.
4. Drop U2 and Pink Floyd themselves and rank the rest.

**Top 10 recommendations**

| Rank | Artist | Score |
|---|---|---|
| 1 | Robbie Williams | 0.394 |
| 2 | Genesis | 0.289 |
| 3 | Johnny Cash | 0.279 |
| 4 | Misfits | 0.250 |
| 5 | Audioslave | 0.241 |
| 6 | Foo Fighters | 0.238 |
| 7 | Pearl Jam | 0.220 |
| 8 | Green Day | 0.217 |
| 9 | Depeche Mode | 0.204 |
| 10 | David Bowie | 0.204 |

Robbie Williams ranks first because he is among the artists most similar to both U2 and Pink Floyd.

### B. User-User CF: recommendations for user 1606
User 1606 has listened to only 4 artists: **ABBA, Elvis Presley, Frank Sinatra, The Beatles**.

1. Compute a 100 × 100 **user-user cosine similarity matrix**.
2. Find the most similar user: **user 1144** (similarity 0.277).
3. Recommend user 1144's artists that user 1606 hasn't heard yet.

**11 recommendations:** Beastie Boys, Bob Dylan, Bob Marley & The Wailers, David Bowie, Eric Clapton, Johnny Cash, Pearl Jam, Pink Floyd, The Doors, The Rolling Stones, Tom Waits

The similarity score is modest because user 1606's listening history is so short.

### C. Overlap check
- After filtering, **0** of the recommendations are artists user 1606 already knows.
- Before filtering, user 1144 and user 1606 shared **2** artists: The Beatles and Elvis Presley.

### D. Hybrid user-item recommendation score
Item-item CF gives everyone who likes U2 the same recommendations, while user-user CF relies on one similar user and ignores how artists relate to each other. The hybrid approach combines both. For each artist user 1606 hasn't heard:

$$\text{score}(a) = \frac{\sum_{j \in \text{top-10 similar}(a)} \text{history}_{1606}(j) \cdot \text{sim}(a, j)}{\sum_{j \in \text{top-10 similar}(a)} \text{sim}(a, j)}$$

**Top 5 hybrid recommendations for user 1606**

| Rank | Artist | Score |
|---|---|---|
| 1 | Eric Clapton | 0.203 |
| 2 | Howard Shore | 0.172 |
| 3 | David Bowie | 0.171 |
| 4 | Dream Theater | 0.135 |
| 5 | Apocalyptica | 0.107 |

Scores are fairly low (max ≈ 0.20) because 4 seed artists don't give much signal.

---

## Part 2: Neural Network Using Only NumPy

### Data
A synthetic binary classification dataset from `sklearn.datasets.make_gaussian_quantiles`: 200 samples, 2 features, 2 classes.

### Architecture
```
Input (2) → Hidden layer (4 units, tanh) → Output (1 unit, sigmoid)
```
- Weights start as small random values (× 0.01); biases start at zero.
- **Cost:** binary cross-entropy. Initial cost ≈ **0.693** (= ln 2, as expected when the model starts out predicting ≈ 0.5 for everything).

### Backpropagation
```
dZ2 = A2 − Y
dW2 = (1/m) · dZ2 · A1ᵀ          db2 = (1/m) · Σ dZ2
dZ1 = (W2ᵀ · dZ2) ⊙ (1 − A1²)     # derivative of tanh
dW1 = (1/m) · dZ1 · Xᵀ           db1 = (1/m) · Σ dZ1
```

### Learning Rate Experiment
The network was trained for 10 gradient descent iterations at three learning rates, tracking how much the output bias `b2` changes at each step.

| Learning rate | Δb2 per step (approx.) |
|---|---|
| 0.0001 | −4.52 × 10⁻¹¹ |
| 0.01 | −4.52 × 10⁻⁹ |
| 1 | −4.52 × 10⁻⁷ → −3.33 × 10⁻⁷ |

**Findings**
- Since Δb2 = −learning rate × dJ/db2, the update grows in direct proportion to the learning rate: each 100× increase in the learning rate gives a 100× bigger step.
- 0.0001 makes negligible updates, so training would be very slow.
- 0.01 makes stable, consistent updates.
- With a learning rate of 1, the step size **shrinks across iterations**. The gradient itself gets smaller as the weights move, which suggests the optimizer is heading toward a minimum despite the large step size.
