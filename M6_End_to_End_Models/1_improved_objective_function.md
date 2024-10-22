Recall from Module 3 that he most common objective function used for training neural networks for classification tasks is frame-based cross entropy. With this objective function, a single one-hot label $z[t]$ is specified for every input frame of data $t$, and compared with the softmax output of the acoustic model. If we define

$$
z[i, t]=\left\{\begin{array}{cc}
1 & z[t]=i \\
0 & \text { otherwise }
\end{array}\right.
$$

then the cross-entropy against the softmax network ouput $y[i, t]$ is as follows.

$$
L=-\sum_{t=1}^T \sum_{i=1}^M z[i, t] \log (y[i, t])
$$


Using a frame-based cross entropy objective function implies three things that are untrue for the acoustic modeling task.
1. That every frame of acoustic data has exactly one correct label.
2. The correct label must be predicted independently of the other frames.
3. All frames of data are equally important.

This module explores some alternative strategies that address these modeling deficiencies.