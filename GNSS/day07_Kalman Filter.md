# day07_Kalman Filter

Kalman经典五公式

**预测方程**
$$
\begin{aligned}
\hat{x}_{k}^{-} &=\Phi_{k-1}\hat{x}_{k-1}	\\
P_{k}^{-} &=\Phi_{k-1}P_{k-1}\Phi_{k-1}^T+Q_{k-1}
\end{aligned}
$$
**更新方程**
$$
\begin{aligned}
K_{k} &=P_{k}^{-}H_{k}^{\mathrm{T}}(H_{k}P_{k}^{-}H_{k}^{\mathrm{T}}+R_{k})^{-1}	\\
\hat{x}_{k} &=\hat{x}_{k}^{-}+K_{k}(y_{k}-H_{k}\hat{x}_{k}^{-})\\
P_{k} &=(I-K_{k}H_{k})P_{k}^{-}
\end{aligned}
$$

$$
x_o,P_0,R_0
$$

$$
\begin{equation*}
\begin{aligned}
\mathbf{\Phi_{k-1}}&=
\begin{bmatrix}
I_{3 \times 3} & 0  &  0 	\\
0 & I_{2 \times 2} & 0	\\
0 & 0 & I_{n \times n}
\end{bmatrix}	\\
\\
\mathbf{Q_{k-1}}&=
\begin{bmatrix}
100 &  & \cdots &  &  & 0	\\
 & 100 & & & &	\\
 \vdots & & 100 & & & \vdots	\\
  &  &  & 10^6 & & 	\\
  &  &  &  & 3 \times 10^{-8} & \\
  0 & & \cdots & & & I_{n \times n}
\end{bmatrix}
\end{aligned}
\end{equation*}
$$

