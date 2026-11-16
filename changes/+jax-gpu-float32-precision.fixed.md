complex64 simulations on the JAX backend are accurate to float32 on NVIDIA GPUs. XLA had computed their contractions in TF32 by default, so states and gradients could be off by up to 1e-2.
