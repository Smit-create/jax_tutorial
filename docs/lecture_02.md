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

# Lecture 2: JIT-compilation, Functional Programming and Gradients

<!-- #region -->
## Just-In-Time (JIT) compilation

### What is JIT compilation?

Just-In-Time (JIT) compilation is a technique used to improve the performance of code at runtime. When a function is JIT compiled, it is transformed from its original form into optimized machine code, allowing it to execute more efficiently. JAX's JIT compiler operates on Python functions, analyzing and optimizing them for improved performance.
The JAX just-in-time (JIT) compiler accelerates logic within functions by fusing linear
algebra operations into a single optimized kernel that the host can
launch on the GPU / TPU (or CPU if no accelerator is detected).

### Benefits of JIT Compilation

- **Faster Execution**: JIT compilation can significantly speed up the execution of numerical code by optimizing it for the underlying hardware.
- **Reduced Overhead**: Compiled functions have lower overhead compared to interpreted Python code, leading to improved performance.
- **Dynamic Optimization**: JIT compilers can adaptively optimize code based on runtime characteristics, leading to better performance in real-world scenarios.

### How to Use JIT Compilation with JAX?

JAX provides the `jit` function, which can be used to JIT compile a Python function.


Let's start with an example
<!-- #endregion -->

```python
import jax
import jax.numpy as jnp
```

```python hide-output=false
def f(x):
    a = 3*x + jnp.sin(x) + jnp.cos(x**2) - jnp.cos(2*x) - x**2 * 0.4 * x**1.5
    return jnp.sum(a)
```

Let’s build an array to call the function on.

```python hide-output=false
n = 50_000_000
x = jnp.ones(n)
```

How long does the function take to execute?

```python hide-output=false
%time f(x).block_until_ready()
```

>**Note**
>
>Here, in order to measure actual speed, we use the `block_until_ready()` method
to hold the interpreter until the results of the computation are returned from
the device. This is necessary because JAX uses asynchronous dispatch, which
allows the Python interpreter to run ahead of GPU computations.

The code doesn’t run as fast as we might hope, given that it’s running on a GPU.

But if we run it a second time it becomes much faster:

```python hide-output=false
%time f(x).block_until_ready()
```

This is because the built in functions like `jnp.cos` are JIT compiled and the
first run includes compile time.

Why would JAX want to JIT-compile built in functions like `jnp.cos` instead of
just providing pre-compiled versions, like NumPy?

The reason is that the JIT compiler can specialize on the *size* of the array
being used, which is helpful for parallelization.

For example, in running the code above, the JIT compiler produced a version of `jnp.cos` that is
specialized to floating point arrays of size `n = 50_000_000`.

We can check this by calling `f` with a new array of different size.

```python hide-output=false
m = 50_000_001
y = jnp.ones(m)
```

```python hide-output=false
%time f(y).block_until_ready()
```

Notice that the execution time increases, because now new versions of
the built-ins like `jnp.cos` are being compiled, specialized to the new array
size.

If we run again, the code is dispatched to the correct compiled version and we
get faster execution.

```python hide-output=false
%time f(y).block_until_ready()
```

The compiled versions for the previous array size are still available in memory
too, and the following call is dispatched to the correct compiled code.

```python hide-output=false
%time f(x).block_until_ready()
```

####  Compiling the outer function

We can do even better if we manually JIT-compile the outer function.

```python hide-output=false
f_jit = jax.jit(f)   # target for JIT compilation
```

Let’s run once to compile it:

```python hide-output=false
f_jit(x)
```

And now let’s time it.

```python hide-output=false
%time f_jit(x).block_until_ready()
```

Note the speed gain.

This is because the array operations are fused and no intermediate arrays are created.

Incidentally, a more common syntax when targetting a function for the JIT
compiler is

```python hide-output=false
@jax.jit
def f(x):
    a = 3*x + jnp.sin(x) + jnp.cos(x**2) - jnp.cos(2*x) - x**2 * 0.4 * x**1.5
    return jnp.sum(a)
```

### Limitations and Considerations

While JIT compilation can improve performance, there are some limitations and considerations to keep in mind:

- **Compilation Overhead**: There may be an initial overhead associated with compiling the function, especially for small functions or infrequently executed code.
- **Memory Usage**: Compiled functions may consume more memory due to the generated machine code.
- **Dynamic Features**: Some Python features, such as dynamic control flow or variable-length data structures, may limit the effectiveness of JIT compilation.

<!-- #region -->
## Functional Programming

From JAX’s documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has “una anima di pura programmazione funzionale”.*

In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure.

A pure function will always return the same result if invoked with the same inputs.

In particular, a pure function has

- no dependence on global variables and  
- no side effects  


JAX will not usually throw errors when compiling impure functions but execution becomes unpredictable.

Here’s an illustration of this fact, using global variables:
<!-- #endregion -->

```python hide-output=false
a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```python hide-output=false
x = jnp.ones(2)
```

```python hide-output=false
f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected — as long as the same compiled version is called.

```python hide-output=false
a = 42
```

```python hide-output=false
f(x)
```

Changing the dimension of the input triggers a fresh compilation of the function, at which time the change in the value of `a` takes effect:

```python hide-output=false
x = jnp.ones(3)
```

```python hide-output=false
f(x)
```

Moral of the story: write pure functions when using JAX!


## Automatic Differentiation

### What is Automatic Differentiation?

Automatic Differentiation (AD) is a technique for efficiently computing derivatives of functions, including gradients and higher-order derivatives, with respect to their inputs. Unlike numerical or symbolic differentiation, AD leverages the chain rule to decompose complex functions into simpler elementary operations.

### Theory Behind Automatic Differentiation

AD computes derivatives by recursively applying the chain rule to decompose a function into elementary operations, known as "primal" and "tangent" evaluations. It can be implemented using two modes: forward-mode AD and reverse-mode AD, each suited for different use cases.

### Gradients


JAX provides the `grad` function to compute gradients of scalar functions. Let's see a simple example:

```python
# Define a function
def f(x):
    return jnp.cos(jnp.sin(x**2))
```

```python
# Compute the gradient of the function
f_prime = jax.grad(f)

print("Gradient of f:", f_prime(10.0))
```

Let’s plot the function and derivative, noting that $ f'(x) = x $.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_grid = jnp.linspace(1, 10, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, [f_prime(x) for x in x_grid], label="$f'$")
ax.legend(loc='upper left')
plt.show()
```

JAX's `grad` function can also compute gradients of functions with multiple arguments. Let's see an example of computing the gradient of a function with respect to multiple variables:

```python
# Define a function of two variables
def g(x, y):
    return jnp.sin(x) * jnp.cos(y)

# Compute the gradient of g with respect to both x and y
g_prime = jax.grad(g, argnums=(0, 1))

# Evaluate the gradient at (x, y) = (0, 0)
result = g_prime(0.0, 0.0)
print("Gradient of g at (0, 0):", result)
```

In this example, `argnums=(0, 1)` specifies that we want to compute the gradient with respect to both the first and second arguments of the function `g`.


#### Using `jax.value_and_grad` for Function and Gradient Computation

JAX provides the `value_and_grad` function to compute both the function value and its gradient simultaneously, improving efficiency. Here's an example:

```python
# Compute the function value and gradient simultaneously
value, grad_value = jax.value_and_grad(f)(jnp.array(3.0))

print("Function value:", value)
print("Gradient of f at x=3.0:", grad_value)
```

### Computing Hessian

The `hessian` function computes the Hessian matrix of the function `f`, which represents the matrix of second-order partial derivatives.

```python
# Compute the Hessian matrix of f
hessian_f = jax.hessian(f)

# Evaluate the Hessian at x = 2
result = hessian_f(2.0)
print("Hessian of f at x=2:", result)
```

The `hessian` function computes the Hessian matrix of the function `f`, which represents the matrix of second-order partial derivatives.


### Computing Jacobians

JAX provides a `jacobian` function to compute the Jacobian matrix of a vector-valued function.

Let's see an example

```python
# Define a vector-valued function
def h(x):
    return jnp.array([x[0] ** 2, jnp.sin(x[1])])

# Compute the Jacobian of h using forward-mode differentiation
jacobian_h = jax.jacfwd(h)

# Evaluate the Jacobian at x = [1, 0]
result = jacobian_h(jnp.array([1.0, 0.0]))
print("Jacobian of h at [1, 0]:", result)
```

<!-- #region -->
## References


- [QuantEcon's JAX introduction](https://jax.quantecon.org/jax_intro.html)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/index.html)
<!-- #endregion -->
