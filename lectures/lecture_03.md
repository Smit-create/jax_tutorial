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

# Loops and Conditions in JAX



## Introduction to Loops in JAX

Loops are pivotal in repetitive tasks, such as iterating over sequences or performing computations iteratively. JAX provides various loop constructs, including `jax.lax.fori_loop`, `jax.lax.while_loop`, and `jax.lax.scan`, enabling fine-grained control over looping mechanisms. In this lecture, we'll delve into these constructs and demonstrate their usage through practical examples.

```{code-cell} ipython3
import jax.numpy as jnp
import jax
```

### Using `jax.lax.fori_loop`

`jax.lax.fori_loop` is a loop construct in JAX that allows for iterating a fixed number of times. It's akin to Python's for loop but optimized for computation within JAX's framework. This function is useful for tasks that require repeated computations or transformations over a predetermined range of iterations.

The `jax.lax.fori_loop` function facilitates a loop with a predetermined number of iterations, similar to Python's `for` loop.

Let's demonstrate its usage with a simple example

```{code-cell} ipython3
# Define a Python loop to sum the squares of numbers from `start` to `end`
def sum_squares(start, end):
    total_sum = 0
    for i in range(10):
        total_sum += i ** 2
    return total_sum
```

```{code-cell} ipython3
sum_squares(1, 10)
```

Now, let's try to re-write the above function using [jax.lax.fori_loop](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html).

In `jax.lax.fori_loop`, the arguments are passed in the following manner:

1. **Start Value**: This argument specifies the initial value of the loop variable.
2. **End Value**: This argument specifies the upper bound for the loop variable. The loop will iterate until the loop variable reaches this value.
3. **Body Function**: This is a function that defines the body of the loop. It takes two arguments: the loop variable and the carry value. The loop variable represents the current iteration index, while the carry value represents any intermediate state that needs to be maintained across loop iterations.
4. **Initial Carry Value**: This argument specifies the initial value of the carry variable, which is passed to the body function in each iteration.

```{code-cell} ipython3
# Rewrite the loop using jax.lax.fori_loop
def sum_squares_jax(start, end):

    def body_fun(i, total):
        return total + i ** 2

    return jax.lax.fori_loop(start,    # lower
                             end,      # upper
                             body_fun, # body_fun
                             0)        # init_val (of total)
```

```{code-cell} ipython3
sum_squares_jax(0, 10)
```

In this example, we define a function `sum_squares_jax` that computes the sum of squares from a given start value to an end value using `jax.lax.fori_loop`. The `body_fun` function squares each number from the loop index `i` and accumulates the result in the `total` variable. Finally, the loop is executed with the specified start and end values, and the result is returned.



### Using `jax.lax.while_loop`

`jax.lax.while_loop` is another looping construct provided by JAX, enabling iterative execution until a termination condition is met. It resembles Python's while loop but is designed to seamlessly integrate with JAX's computational graph and automatic differentiation capabilities. while_loop is suitable for situations where the number of iterations is not known beforehand and depends on runtime conditions.

Let's illustrate its usage with an example

```{code-cell} ipython3
# Define a Python while loop to compute the factorial of `n`
def factorial(n):
    result = 1
    i = 1
    while i <= n:
        result *= i
        i += 1
    return result

factorial(6)
```

Now, let's try to re-write the above function using [jax.lax.while_loop](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html)


In `jax.lax.while_loop`, the arguments are passed as follows:

1. **Loop Condition Function**: This function defines the termination condition of the loop. It takes the current loop state as its argument and returns a boolean value indicating whether the loop should continue (`True`) or terminate (`False`).
2. **Loop Body Function**: This function defines the body of the loop. It takes the current loop state as its argument and returns the updated loop state for the next iteration.
3. **Initial Loop State**: This argument specifies the initial state of the loop, which is passed to both the loop condition and loop body functions.

```{code-cell} ipython3
# Rewrite the loop using jax.lax.while_loop
def factorial_jax(n):

    def condition(state):
        i, result = state
        return i <= n

    def body(state):
        i, result = state
        return (i + 1, result * i)

    _, result = jax.lax.while_loop(condition, # cond_fun
                                   body,      # body_fun
                                   (1, 1))    # init_value (i=1, result=1)
    return result
```

```{code-cell} ipython3
factorial_jax(6)
```

In this example, we define a function `factorial_jax` that computes the factorial of a number using `jax.lax.while_loop`. The `condition` function checks if the loop variable `i` is less than or equal to `n`, while the `body` function updates the loop state by incrementing `i` and accumulating the factorial in the `result` variable. The loop continues until the condition is `False`, and the final result is returned.

Since in the final result we get the value of `(i, result)`, we ignore the first value
and return the result.



### Using `jax.lax.scan`

`jax.lax.scan` is a function in JAX for performing cumulative computations over a sequence of inputs. It's similar to Python's accumulate function but optimized for efficient execution within JAX's framework. `scan` is commonly used for tasks such as computing cumulative sums, products, or applying a function iteratively over a sequence while accumulating results. It's a powerful tool for implementing recurrent neural networks, sequential models, or any computation involving cumulative operations.

`jax.lax.scan` is generalized version of handling loops in JAX and can handle complex looping constructs.

Let's see the following example

```{code-cell} ipython3
# Define a Python function to compute cumulative sums of a list
def cumulative_sums(nums):
    cumulative_sums = []
    total = 0
    for num in nums:
        total += num
        cumulative_sums.append(total)
    return cumulative_sums
```

```{code-cell} ipython3
nums = [1, 2, 3, 4, 5]
cumulative_sums(nums)
```

Now, let's try to re-write the above function using [jax.lax.scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)


In `jax.lax.scan`, the arguments are passed as follows:

1. **Body Function**: This function defines the computation to be performed at each step of the loop. It takes two arguments: the loop variable (or current input element) and the carry variable (or accumulated state), and returns a tuple containing the updated loop variable and the updated carry variable.
2. **Initial Carry Value**: This argument specifies the initial value of the carry variable, which is passed as the initial state to the loop.
3. **Sequence**: This argument specifies the input sequence over which the loop iterates.

```{code-cell} ipython3
# Rewrite the computation using jax.lax.scan
def cumulative_sums_jax(nums):

    def body(total, num):
        return total + num, total + num

    total, cumulative_sums_array =  jax.lax.scan(body,   # f
                                                 0,      # init
                                                 nums)   # xs
    return cumulative_sums_array
```

```{code-cell} ipython3
cumulative_sums_jax(jnp.array(nums))
```

In this example, we define a function `cumulative_sums_jax` that computes cumulative sums using `jax.lax.scan`. The `body` function computes the sum of the current element and the carry variable, updating both the loop variable and the carry variable. The loop iterates over the input sequence, accumulating the sums at each step, and the final result is returned.



## Conditional Execution with JAX

### Introduction to `jax.lax.cond`

`jax.lax.cond` is a conditional execution function provided by JAX, allowing users to perform different computations based on specified conditions. This enables dynamic control flow within JAX computations, facilitating conditional branching similar to Python's `if` statement. We'll explore the usage of `jax.lax.cond` through practical examples.

```{code-cell} ipython3
# Define a Python function to check if a number is positive or negative
def check_sign(x):
    if x > 0:
        return 1
    else:
        return -1

# Execute the Python function with a sample input
print("Sign of 5 (Python):", check_sign(5))
print("Sign of -10 (Python):", check_sign(-10))
```

Let's re-write the same using [jax.lax.cond](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html).

In `jax.lax.cond`, the arguments are passed as follows:

1. **Predicate**: This is a boolean scalar indicating the condition to be evaluated. If the predicate is `True`, the `true_fun` will be executed; otherwise, the `false_fun` will be executed.
2. **True Function**: This function defines the computation to be performed if the predicate is `True`. It takes no arguments and returns the result of the computation when the condition is satisfied.
3. **False Function**: This function defines the computation to be performed if the predicate is `False`. It takes no arguments and returns the result of the computation when the condition is not satisfied.

```{code-cell} ipython3
# Rewrite the function using jax.cond
def check_sign_jax(x):
    def positive_branch(x):
        return 1
    def negative_branch(x):
        return -1

    return jax.lax.cond(x > 0,           # pred
                        positive_branch, # true_fn
                        negative_branch, # false_fn
                        x)               # operands

# Execute the JAX function with the same input
print("Sign of 5 (JAX cond):", check_sign_jax(5))
print("Sign of -10 (JAX cond):", check_sign_jax(-10))
```

In this example, we define a function `check_sign_jax` that checks if a number is positive or negative using `jax.lax.cond`. Depending on whether the input `x` is greater than 0 (positive) or not (negative), the corresponding true or false function is executed, and the result is returned.



## Why do we need `jax.lax` ?



While JAX provides high-level abstractions for numerical computing, leveraging low-level constructs from `jax.lax` can lead to significant speedups, especially when compared to traditional Python for loops.

Moreover, in-order to use the JAX's JIT, sometime its necessary to leverage low-level constructs.

###  Importance of Performance Optimization

Efficient computation is essential for tackling complex problems in machine learning, scientific computing, and other domains. Performance optimization techniques, such as minimizing computational overhead and maximizing hardware utilization, are critical for achieving faster execution times and scaling to larger datasets or models.

Let's take the example from `fori_loop` section and
form a compiled jit functions

```{code-cell} ipython3
computation_jit_lax = jax.jit(sum_squares_jax)
computation_jit_python = jax.jit(sum_squares, static_argnums=(0, 1))

# Compare execution times
x = 10000
```

For `computation_jit_python`, JAX requires that the `static_argnums` parameter be provided because `range` itself is a dynamic operation. By specifying the index of the argument that corresponds to the `range`'s upper bound in `static_argnums`, JAX can treat the `range` as static during compilation, optimizing the loop accordingly. This helps avoid unnecessary recompilation of the loop body for different loop bounds, leading to improved performance.

```{code-cell} ipython3
%time result_jit_lax = computation_jit_lax(0, x)
```

```{code-cell} ipython3
%time result_jit_lax = computation_jit_lax(0, x)
```

Running the `computation_jit_lax` takes a bit more time in the first call because of the compilation overhead

```{code-cell} ipython3
%time result_jit_python = computation_jit_python(0, x)
```

```{code-cell} ipython3
%time result_jit_python = computation_jit_python(0, x)
```

Notice the time difference in 2nd calls of both `computation_jit_lax` and `computation_jit_python`.

The function `computation_jit_lax` has clear advantanges over `computation_jit_python` because of two major reasons:
- Very fast because of `jax.lax` and low-level optimizations done by jax.
- `computation_jit_python` has used `static_argnums` in `jax.jit` which means for every new values of `start` and `end`, `computation_jit_python` will re-compile and evaluate the results which will make it even slower unlike `computation_jit_lax`. Once compiled, `computation_jit_lax` will call the same function irrespective of the value of `start` and `end`.

## References


- [JAX Documentation](https://jax.readthedocs.io/en/latest/index.html)
