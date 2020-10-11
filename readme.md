# vLin: Vectorised Linear Modelling

This package is an experiment to try to vectorise (a la numpy) the process 
of constructing linear models. It works, but is **marginally slower than 
[python-mip](https://github.com/coin-or/python-mip)**
for problems with ~300 variables and ~1000 constraints.

### Goal

- First-class support for linear expressions, including indexing and creating
  constraints.
- High speed model construction,  perhaps at the expense of being easy or
  safe to use
 
- Extremely clear and reasonably terse construction of models
  (through operator overloading "<=", ">=", "==" for constraints, and "+=" for
  adding constraints to model)
- Easy debugging
  
### Design

The key data structure in this program is an expression which is
a constant plus a linear combination of variables from the model.
For example, an expression might be ``1.3 + 2.1*var_1 + 0.2*var_3``.

A single expression is represented abstractly as a vector where the first element
is the constant and all other elements are the coefficients of the variables
in the model. For example, ``1.3 + 2.1*var_1 + 0.2*var_3`` is represented by
``[1.3 2.1 0 0.2 0 0 0 0 ...]``.

A number of these single expressions are combined together into a matrix
to form a vector of expressions - that is, a matrix where each row is
a single expression; a vector of expressions. This is what 
the ``Expr`` class in the code represents.

A concrete example:

```python
>>> import vlin
>>> m = vlin.Model()
>>> a = m.var(3)
>>> a
ExprNumpy([[0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.]])
>>> b = a + np.array([1, 2, 3])
>>> b
ExprNumpy([[1., 1., 0., 0., 0., 0.],
           [2., 0., 1., 0., 0., 0.],
           [3., 0., 0., 1., 0., 0.]])
>>> a + b
ExprNumpy([[1., 2., 0., 0., 0., 0.],
           [2., 0., 2., 0., 0., 0.],
           [3., 0., 0., 2., 0., 0.]])
```

This format is conveniently very close to the standard form a linear program.

Constraints are represented as ``Expr`` objects which must be ``<=0``.
### Post-mortem

It's not very fast at all. But, at least it's not super slow on smaller problems.
Two classes of ``Expr`` were implemented: one based on dense numpy arrays,
and one based on sparse scipy matrices. 

I guess all the redundant operations from numpy added substantial overhead,
and the sparse matrices themselves have substantial overhead.

### Usage

This is a dead project, a failed experiment, a stepping stone to something better.

But yeah if you really want to use it check out the tests for a couple of small
examples.  Perhaps this project could be useful for educational purposes.
Don't expect it to be robust.

### Next steps

How to make linear model construction in Python fast?

Just use C?

Just use JuMP?

Just use PyPy?

There must be a better way.

### Driver

Why make this? Good question, there are quite a few linear modelling
packages for python. 

For a start, there's always value in 
[reinventing the wheel](https://sanpv.com/2019/11/02/the-hidden-value-of-reinventing-the-wheel/)

For many problems the definition of the model - as opposed
to actually solving the model - takes the vast majority of the time.
I want first-class support for linear expressions.
No package could satisfy me - these are my hot takes:

- python-mip
  - **Specifically:** Can't do vectorisation, and repeated instance checks 
  really slow things down. 
  - **In general:** overall a very nice package, clean code, but could
    do with a bit more of a focus on rigorous testing.
- cylp
  - **Specifically:** Can't index into arrays of linear expressions. 
  - **In general:** Poor docs and arcane/complex code and object interactions.
- cvxpy
  - **Specifically:** horrendously slow "compilation" step between 
  defining and solving model
  - **In general:** Can do nonlinear opt so might be a good choice if you
    want to leave that as an option.
- pulp
  - **Specifically:** Can't do vectorisation, not tightly integrated with
    solvers.
  - **In general:** haven't used it much
- pyomo
  - **Specifically:** Slow to run repeated optimisations.
  - **In general:** Extremely verbose modelling language