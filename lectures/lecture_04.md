---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Vectorization and Parallelization in JAX

## Introduction to Vectorization

Vectorization is a technique used in numerical computing to perform operations on entire arrays or batches of data simultaneously, leveraging the inherent parallelism of modern hardware. By operating on vectors or arrays rather than individual elements, vectorized code can achieve significant performance improvements, especially when dealing with large datasets or computational tasks. In JAX, vectorization plays a crucial role in optimizing computations for efficiency and scalability.

Writing fast JAX code requires shifting repetitive tasks from loops to array processing operations, so that the JAX compiler can easily understand the whole operation and generate more efficient machine code.

This procedure is called **vectorization** or **array programming**, and will be familiar to anyone who has used NumPy.

In most ways, vectorization is the same in JAX as it is in NumPy.

```{code-cell} ipython3
import jax.numpy as jnp
import jax
import numpy as np
```

### Writing Vectorized Code in JAX

Vectorized code in JAX can be written using JAX's NumPy-like API, which supports operations on arrays and tensors. By expressing computations in terms of array operations, such as element-wise arithmetic, matrix multiplication, and broadcasting, developers can harness the full potential of vectorization in JAX.

Let's illustrate this with a simple example

```{code-cell} ipython3
# Define a vectorized computation to calculate the element-wise square of an array
def square_elements(x):
    return x ** 2

# Apply the vectorized computation to an array
input_array = jnp.array([1, 2, 3, 4, 5])
result_array = square_elements(input_array)
print("Squared elements:", result_array)
```

But there are also some differences, which we highlight here.

As a running example, consider the function

$$
f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
$$

Suppose that we want to evaluate this function on a square grid of $ x $ and $ y $ points and then plot it.

To clarify, here is the slow `for` loop version.

```{code-cell} ipython3
:hide-output: false

@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)

n = 80
x = jnp.linspace(-2, 2, n)
y = x

z_loops = np.empty((n, n))
```

```{code-cell} ipython3
:hide-output: false

%%time

for i in range(n):
    for j in range(n):
        z_loops[i, j] = f(x[i], y[j])
```

Even for this very small grid, the run time is extremely slow.

(Notice that we used a NumPy array for `z_loops` because we wanted to write to it.)

OK, so how can we do the same operation in vectorized form?

If you are new to vectorization, you might guess that we can simply write

```{code-cell} ipython3
:hide-output: false

z_bad = f(x, y)
```

But this gives us the wrong result because JAX doesn’t understand the nested for loop.

```{code-cell} ipython3
:hide-output: false

z_bad.shape
```

Here is what we actually wanted:

```{code-cell} ipython3
:hide-output: false

z_loops.shape
```

To get the right shape and the correct nested for loop calculation, we can use a `meshgrid` operation designed for this purpose:

```{code-cell} ipython3
:hide-output: false

x_mesh, y_mesh = jnp.meshgrid(x, y)
```

Now we get what we want and the execution time is very fast.

```{code-cell} ipython3
:hide-output: false

%%time

z_mesh = f(x_mesh, y_mesh)
```

```{code-cell} ipython3
:hide-output: false

%%time

z_mesh = f(x_mesh, y_mesh)
```

Let’s confirm that we got the right answer.

```{code-cell} ipython3
:hide-output: false

jnp.allclose(z_mesh, z_loops)
```

Now we can set up a serious grid and run the same calculation (on the larger grid) in a short amount of time.

```{code-cell} ipython3
:hide-output: false

n = 6000
x = jnp.linspace(-2, 2, n)
y = x
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

```{code-cell} ipython3
:hide-output: false

%%time

z_mesh = f(x_mesh, y_mesh)
```

```{code-cell} ipython3
:hide-output: false

%%time

z_mesh = f(x_mesh, y_mesh)
```

But there is one problem here: the mesh grids use a lot of memory.

```{code-cell} ipython3
:hide-output: false

x_mesh.nbytes + y_mesh.nbytes
```

By comparison, the flat array `x` is just

```{code-cell} ipython3
:hide-output: false

x.nbytes  # and y is just a pointer to x
```

This extra memory usage can be a big problem in actual research calculations.

### Using `jax.vmap`

So let’s try a different approach using [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)

#### Converting loops to `jax.vmap`

The process of converting loops to `jax.vmap` is simple. You take the inner most loop
and vectorized your code over this loop and start forming a nested sequence of `jax.vmap`
from inner most loop to outer most.

Here in our example, first we vectorize `f` in `y` because `y` is in the inner loop.

```{code-cell} ipython3
:hide-output: false

f_vec_y = jax.vmap(f, in_axes=(None, 0))
```

In the line above, `(None, 0)` indicates that we are vectorizing in the second argument, which is `y`.

Next, we vectorize in the first argument, which is `x` and used in the outer loop using
`f_vec_y`.

```{code-cell} ipython3
:hide-output: false

f_vec = jax.vmap(f_vec_y, in_axes=(0, None))
```

With this construction, we can now call the function $ f $ on flat (low memory) arrays.

```{code-cell} ipython3
:hide-output: false

%%time

z_vmap = f_vec(x, y)
```

```{code-cell} ipython3
:hide-output: false

%%time

z_vmap = f_vec(x, y)
```

```{code-cell} ipython3
z_vmap.shape
```

The execution time is essentially the same as the mesh operation but we are using much less memory.

And we produce the correct answer:

```{code-cell} ipython3
:hide-output: false

jnp.allclose(z_vmap, z_mesh)
```

#### Benefits of Vectorization with jax.vmap

Vectorizing computations with `jax.vmap` offers several benefits:

- **Improved Performance**: Vectorized code can leverage hardware parallelism for faster execution, especially on accelerators like GPUs and TPUs.
- **Simplified Code**: By replacing nested loops with array operations, vectorized code tends to be more concise, readable, and maintainable.
- **Scalability**: Vectorization enables scaling computations to larger datasets or batch sizes without significant performance degradation, making it suitable for machine learning tasks and scientific computing.
- **Nested Vmaps**: Once can easily replace a sequence of python loops using a sequence of `jax.vmap` and JAX handles all the optimizations.

+++

## Introduction to parallelization

Parallelization is a crucial technique for accelerating computations by leveraging the parallelism of modern hardware. In JAX, parallelization can be achieved using `jax.pmap` for parallel mapping and `jax.pjit` for parallel just-in-time compilation. This lecture will explore these constructs in detail, demonstrating how they enable efficient parallel execution on multi-core CPUs and GPUs.

### Parallel Mapping with `jax.pmap`

`jax.pmap` is a higher-order function in JAX that parallelizes a function across multiple devices or processors. It maps the function over the leading axis of the input arrays and executes the function in parallel on each device. This enables efficient parallel execution of computations, particularly on GPUs or multi-core CPUs.

#### Using `jax.pmap`

Let's illustrate the usage of [jax.pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) with a simple example:

```{code-cell} ipython3
# Define a function to square each element of an array
def square(x):
    return x ** 2

# Parallelize the function using jax.pmap
parallel_square = jax.pmap(square)
```

```{code-cell} ipython3
# Apply the parallelized function to an array
num_available_devices = jax.local_device_count()

input_array = jnp.arange(num_available_devices)

result_array = parallel_square(input_array)
print("Squared elements (parallel):", result_array)
```

The output is just `[0]` because `num_available_devices` on my device is just 1. In case
you are working on a machine having large number of GPUs/TPUs you can easily use `jax.pmap`
to parallelize your execution. The only note while using `pmap` is that:

The mapped `axis` size must be less than or equal to the number of local XLA devices available, as returned by `jax.local_device_count()` (unless devices is specified, see below). For nested `pmap` calls, the product of the mapped axis sizes must be less than or equal to the number of XLA devices.

+++

### Benefits of Parallelization with JAX

- **Performance**: Parallelization enables efficient utilization of hardware resources, leading to faster execution times for computations.
- **Scalability**: With parallel execution, computations can scale to larger datasets and models, accommodating the growing demands of machine learning and scientific computing tasks.
- **Hardware Utilization**: JAX maximizes hardware utilization by leveraging multi-core CPUs and GPUs for parallel execution, improving overall efficiency and productivity.

## References


- [QuantEcon's JAX introduction](https://jax.quantecon.org/jax_intro.html)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/index.html)
