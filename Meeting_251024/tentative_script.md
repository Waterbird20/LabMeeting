## Slide 1

Breif intro to topic

## Slide 2

Breif intro to today's outline.
That is, 
I will explain what DDrf is very breifly(that is will not provide detailed explanation or derivation of equations) and how it can be used for realizing GHZ state. 
I first considered Machine Learning approach that is setting parameters so that we can acheive high fidelity to GHZ state, but it was out of purpose and hard to do.
So the following contents are my strategy with detailed contents.

> outline contents with bullet points

## Slide 3

I'll explain NV center system with Hamiltonian(equation) with one figure(half right)

> The spatially separated magnetic dipoles interact to each other by magnetic dipolar interaction(i.e. magnetic dipole-dipole interaction), which dominates the hyperfine coupling between the NV-'s electron spin and surrounding 13C's nuclear spin. Note: The nuclear spin's precession axes are dependent on the NV-electron's spin state.
> Hamiltonian equation
> Figure for NV center system located at right side with 50% width

## Slide 4

I'll explain the available control source. Microwave for electron spin and RF for nuclear spin. Here, RF can not perform selective nuclear spin(as will be shown in control Hamiltonain)

> Control Hamiltonian will be shown. 




## Slide 5

I'll then explain how Machine Learning approach(Strictly, it is just using Gradient Descent Method) can works. What we are going to do is obtaining GHZ state after time evolution. That is solving time-dependent Schrodinger equation and select optimal params such that the state after time evolution is GHZ state. So, we can define it as machine learning problem where cost function is fidelity of final state to GHZ state. 

> When CPMG based MW and RF operation is available, the full hamiltonian equation is shown. 
> Cost function is shown as equation.

## Slide 5

I'll explain why it is not working. There are few issues. 
1. mere GRAPE algorithm, finding the sequence of time-ordered value of each parameter, is unavailable. The time for DDrf sequence which is known to be able to implement CRX gate is order of micro-second or even mili-second. However, control interval is order of nano-second.
2. Even employing known DDrf form pulse sequence, we have no idea for optimal time. Note: as the number of $\pi$-pulse is discrete, the total time evolution time should be selected explitcitly. 
3. Gradient Descent Methods try to exploit cost function, that is, unable to consider realistic conditions.

> The issues are listed with bullet point.

## Slide 6

A little details for 3rd issue will be discussed. Especially regarding to $\Omega_{RF}$, RF Amplitude. It should not be large, theoritically, large amplitude make unwanted detuning to other nuclear spin. Experimentally, large amplitude can burn the wire. But enforcing this condition to learning is not straightforward. 



> - Qutip's solver can be integrated with scipy solver. How about adding Bounds?

> Q: Why not use contrainted(adding bounds) solver?
> A: I tried but it was bad! I used L-BFGS-B optimizer but failed and takes too long time and failed to converge finally. 

> Q: How about regularization method to avoid L-BFGS-B method. 
> $$
> f_r(\Omega_{RF}) = \Omega_{RF}^2
> $$
> Then 
> $$
> S = f + \lambda f_r
> $$

> A: scipy.optimizer failed. Can not converge. I can not debug builtin optimizer. 

## Slide 7

Now lets build stratgy. 
I'll use Machine Learning Method for fine-tuneing purpose. So I need to provide good parameteres beforehand. So I have to do some analytical analysis to find good params(Jiwon did a lot!).
To be further, I wanna use better optimizers embedded in Pytorch(specifically Adam has better convergence). I personally experienced this when trying to implement GRAPE algorith, which works better with Adam. So, I need new Schrodinger Equation Solver implemented by Pytorch. 
Note: For general purpose, BFGS is better for fine-tuning. But not sure how good parameter we expect to obtain analytically. 

## Slide 8

So, I dedicated about 3 months to study NV system control and implementing new solver. Today I'll mainly discuss about this. First I'll tell you about how we can solve Differential Equation. 

> Basically, it is solving time dependent Schrodinger equation. 

> can we do this like:
> $$
> \ket{\psi(t + dt)} = \ket{\psi(t)} -i H(t) dt
> $$
> (general idea of Newton method.)

## Slide 9

norm preserving issue arised. Error proportional to $O(dt^2)$ in this case. 
I employed Runge Kutta(7-th order). 

## Slide 10

RK explanation

## Slide 12-15

Key features of my implementation.

key features:
1. interpolation.
2. norm preserving check
3. Adaptive time step

## Slide 16

I'll explain Dynamic Decoupling. Original Purpose is Extending coherence time of the spin, but can be used many purpose, Here, I'll show conditional gate implemented by CPMG.

As noted in the first section, nuclear spin precession is dependent on NV-electron's spin state.

> Equation for Hamiltonian

## Slide 17

CPMG is repeat of (\tau - \pi - 2\tau - \pi - \tau) $N/2$ times. The unitary operations where initial electron spin state is 0(1) can be written as $e^{-i \phi \hat{I}\cdot\sigma^{i}}$. When the vector $\sigma^0$ and $\sigma^1$ is anti-parallel, the action of CMPG unit becomes conditional gate. The direction of each vector is generally parallel, however, at a specific $\tau$ becomes anti-parallel. 

## Slide 18

$\tau$ can be found analytically under approximation. 

## Slide 20

At a certain $\tau$, we see conditional operation.
Since assumption is highly short pi-pulse time, further study required. 

> gif file will be shown.

## Slide 21

In a multi qubit system, one of nuclear spin is responsive to the CPMG sequence. 
Further analysis is required.

> gif file will be shown.

## Slide 21

RF pulse can be employed to add additional and active operation on nuclear spin. In the Lab frame, it acts as if oscillating effect to precession axis. If we choose, RF phase update meticulously, we can derive conditional operation(CRX gate).

> For DDRF gates, the interpulse delays do not need to follow the dynamics of the target nuclear spin: Direct spin-state selective radio-frequency driving with tailored phase updating enables a conditional rotation of the nuclear spin.

## Slide 22

> gif for NV spin + 1 nuclei spin will shown

## Slide 23

Multi-qubit simulation will be shown, but further study required to choose good parameters. 

## Slide 24

Todo.
1. Analytical study: Find better parameter
2. polish code: memory management + find better hyper-params for faster simulation + make simulation time in 1min(for learning purpose)
3. Run Learning after obtaining good params
4. (later) Need to develop other solver in pyTorch for density matrix. As NV/nuclei spins are not fully initiallized to pure state. So we need to consider mixed state, leading to density matrix approach. 