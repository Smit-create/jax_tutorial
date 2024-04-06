---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Lecture 1: An Introduction to JAX

JAX is an open-source library developed by Google Research, aimed at numerical computing and machine learning. It provides composable transformations of Python functions, enabling automatic differentiation, efficient execution on accelerators like GPUs and TPUs, and seamless interoperability with NumPy. With its functional programming model and powerful features, JAX offers a versatile platform for high-performance computing tasks.


## JAX as a NumPy Replacement

One way to use JAX is as a plug-in NumPy replacement. Let’s look at the
similarities and differences.


### Similarities

The following import is standard, replacing `import numpy as np`:

```python hide-output=false
import jax
import jax.numpy as jnp
```

Now we can use `jnp` in place of `np` for the usual array operations:

```python hide-output=false
a = jnp.asarray((1.0, 3.2, -1.5))
```

```python hide-output=false
print(a)
```

```python hide-output=false
print(jnp.sum(a))
```

```python hide-output=false
print(jnp.mean(a))
```

```python hide-output=false
print(jnp.dot(a, a))
```

However, the array object `a` is not a NumPy array:

```python hide-output=false
a
```

```python hide-output=false
type(a)
```

Even scalar-valued maps on arrays return JAX arrays.

```python hide-output=false
jnp.sum(a)
```

JAX arrays are also called “device arrays,” where term “device” refers to a
hardware accelerator (GPU or TPU).

(In the terminology of GPUs, the “host” is the machine that launches GPU operations, while the “device” is the GPU itself.)

Operations on higher dimensional arrays are also similar to NumPy:

```python hide-output=false
A = jnp.ones((2, 2))
B = jnp.identity(2)
A @ B
```

```python hide-output=false
from jax.numpy import linalg
```

```python hide-output=false
linalg.inv(B)   # Inverse of identity is identity
```

```python hide-output=false
linalg.eigh(B)  # Computes eigenvalues and eigenvectors
```

### Differences

One difference between NumPy and JAX is that JAX currently uses 32 bit floats by default.

This is standard for GPU computing and can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.  In these cases 64 bit floats can be enforced via the command

```python hide-output=false
jax.config.update("jax_enable_x64", True)
```

Let’s check this works:

```python hide-output=false
jnp.ones(3)
```

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.

For example, with NumPy we can write

```python hide-output=false
import numpy as np
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```python hide-output=false
a[0] = 1
a
```

In JAX this fails:

```python hide-output=false
a = jnp.linspace(0, 1, 3)
a
```

```python hide-output=false
a[0] = 1
```

In line with immutability, JAX does not support inplace operations:

```python hide-output=false
a = np.array((2, 1))
a.sort()
a
```

```python hide-output=false
a = jnp.array((2, 1))
a_new = a.sort()
a, a_new
```

The designers of JAX chose to make arrays immutable because JAX uses a
functional programming style.  More on this below.

Note that, while mutation is discouraged, it is in fact possible with `at`, as in

```python hide-output=false
a = jnp.linspace(0, 1, 3)
id(a)
```

```python hide-output=false
a
```

```python hide-output=false
a.at[0].set(1)
```

We can check that the array is mutated by verifying its identity is unchanged:

```python hide-output=false
id(a)
```

## Random Numbers

Random numbers are also a bit different in JAX, relative to NumPy.  Typically, in JAX, the state of the random number generator needs to be controlled explicitly.

```python hide-output=false
import jax.random as random
```

First we produce a key, which seeds the random number generator.

```python hide-output=false
key = random.PRNGKey(1)
```

```python hide-output=false
type(key)
```

```python hide-output=false
print(key)
```

Now we can use the key to generate some random numbers:

```python hide-output=false
x = random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```python hide-output=false
random.normal(key, (3, 3))
```

To produce a (quasi-) independent draw, best practice is to “split” the existing key:

```python hide-output=false
key, subkey = random.split(key)
```

```python hide-output=false
random.normal(key, (3, 3))
```

```python hide-output=false
random.normal(subkey, (3, 3))
```

The function below produces `k` (quasi-) independent random `n x n` matrices using this procedure.

```python hide-output=false
def gen_random_matrices(key, n, k):
    matrices = []
    for _ in range(k):
        key, subkey = random.split(key)
        matrices.append(random.uniform(subkey, (n, n)))
    return matrices
```

```python hide-output=false
matrices = gen_random_matrices(key, 2, 2)
for A in matrices:
    print(A)
```

One point to remember is that JAX expects tuples to describe array shapes, even for flat arrays.  Hence, to get a one-dimensional array of normal random draws we use `(len, )` for the shape, as in

```python hide-output=false
random.normal(key, (5, ))
```

<!-- #region -->
In conclusion, this introductory lecture has provided a glimpse into the capabilities and features of JAX. We've explored how JAX offers a powerful platform for numerical computing and seamless interoperability with NumPy as well as how it differs from NumPy.

## References


- [QuantEcon's JAX introduction](https://jax.quantecon.org/jax_intro.html)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/index.html)
<!-- #endregion -->