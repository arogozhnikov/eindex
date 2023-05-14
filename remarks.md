Different remarks about eindex, that seem to be too narrow for documentation.


## Inner and outer indexing

Let's learn what is inner and outer indexing by taking pixels of a grayscale `image` with two 1-d arrays `T` and `L`.

Arrays T, L enumerate top and left coordinates for some patch in an image.


```python
# Pseudocode for inner indexing (T and L should be of same length):
result[i] = image[T[i], L[i]] for every i
# Outer indexing is more like independent indexing, so T and L can be of different size
result[t, l] = image[T[t], L[l]] for every t, l

# Outer can be replaced with just a couple of 1d indexers in numpy:
result = image[I, :][:, J]
```

Because outer indexing can be replaced with several indexing operations, eindex focuses on inner indexing.

But there are cases when one needs to mix inner and outer indexing. This can be achieved with dummy axes. 

We continue previous example, but this time we have a number of images, and we want to collect a single patch from every position:

```images_bhwc, T_bt, L_bl```

we broadcast indices to the same order btl:

```python

patches = gather(
    'b t l c <- b T L c, [T, L] b t l', images, 
    # T_bt -> T_bt1, L_bl -> L_b1l
    [T_bt[:, :, None], L_bl[:, None, :]]
)
```

Note: `np.ufunc.at` also accepts tuple of indexers, and broadcasts them.
Eindex makes broadcasting (i.e. you can use dummy axes, and combined index will be computed efficiently),
BUT eindex requires that number of dimensions is the same.

That's just a safety thing.


## Einindex

einindex is @malmaud's prototype of indexing (supports only 1d and 2d indexing for torch)

Einindex's interface is very similar to eindex.gather with some differences:

1. indexer dimension is last, not first
2. 1d and 2d are not consistent. 1d is exceptional because an additional dimension is not 


## Why commas inside square brackets ` [a, b, c] d e f`?

- space-separated dimensions in einops are just different axes, so better avoid clashing with it
- this better mirrors ability to get/pass lists in python, e.g.

```python
[a, b, c] = argmax(x, 'a b c d e f -> [a,b,c] d e f')
```