<div align="center">
  <img src="https://arogozhnikov.github.io/images/eindex/logo-animated.gif" alt="eindex animated logo" />
</div>

# eindex

Concept of multidimensional indexing for tensors


## Example of K-means clustering


Plain numpy

```python
def kmeans(init_centers, X, n_iterations: int):
    n_clusters, n_dim = init_centers.shape
    n_onservations, n_dim = X.shape

    centers = init_centers.copy()
    for _ in range(n_iterations):
        d = cdist(centers, X)
        clusters = np.argmin(d, axis=0)
        new_centers_sum = np.zeros_like(centers)
        indices_dim = np.arange(n_dim)[None, :]
        np.add.at(new_centers_sum, (clusters[:, None], indices_dim), X)
        cluster_counts = np.bincount(clusters, minlength=n_clusters)
        centers = new_centers_sum / cluster_counts[:, None]
    return centers
```


With eindex

```python
def kmeans_eindex(init_centers, X, n_iterations: int):
    centers = init_centers
    for _ in range(n_iterations):
        d = cdist(centers, X)
        clusters = EX.argmin(d, 'cluster i -> [cluster] i')
        centers = EX.scatter(X, clusters, 'i c, [cluster] i -> cluster c',  
                             agg='mean', cluster=len(centers))
    return centers
```

## [Tutorial notebook](https://github.com/arogozhnikov/eindex/blob/main/tutorial/tutorial.ipynb)


## Goals

- Form helpful 'language' to think about indexing and index-related operations. Tools shape minds 
- Cover most common cases of multidimensional indexing that are hard to implement using the standard API
- Approach should be applicable to most common tensor frameworks, autograd should work out-of-the-box
- Aim for readable and reliable code
- Allow simple adoption in existing codebases
- Implementation should base on fairly common tensor operations. No custom kernels allowed.
- Complexity should be visible: execution plan for every operation should form a static graph. 
  Structure of the graph depends on the pattern, but not on tensor arguments.

Non-goals: there is no goal to develop 'the shortest notation' or 'the most advanced/comprehensive tool for indexing' or 'cover as many operations as possible' or 'completely replace default indexing'.



## Examples

Follow [tutorial](https://github.com/arogozhnikov/eindex/blob/main/tutorial/tutorial.ipynb) first to learn about all operations provided.

<details>
<summary>Click to unfold</summary>

#### - how do I select a single embedding from every image in a batch?

Let's say you have pairs of images and captions, and you want to take closest embedding:
```python
score = einsum(images_bhwc, sentences_btc, 'b h w c, b token c -> b h w token')
closest_index = argmax(score, 'b h w token -> [h, w] b token')
closest_emb = gather('b h w c, [h, w] b token -> b t c', images_bhwc, closest_index)
```

To adjust this example for video not image, replace 'h w' to 'h w t'. Yes, that simple.


#### - how to collect top-1 or top-3 predicted word for every position in audio/text?

```python
[most_likely_words] = argmax(prob_tbc, 't b w -> [w] t b')
[top_words] = argsort(prob_tbc, 't b w -> [w] t b order')[..., -3:]
```

#### - how to average embeddings over neighbors in a graph?
```python
# without batch (single graph)
gather('vin c, [vin, vout] edge -> vout', embeddings, edges)
# with batch (multile graphs)
gather('b vin c, [b, vin, vout] edge -> b vout', embeddings, edges)
``` 

#### - can eindex help with (complex) positional embeddings?

If we're speaking about trainable abspos, it can be just saved as `emb_hwc` and added every time to a batch.
There is no need for indexing. 

But it can be very helpful for complex scenarios: for example in T5-relpos, when a bias is added to every logit before softmax-ing to compute attention?
That's simple to implement for 1d, and *much* harder for 2d/3d. Let's implement for 2d with eindex:
```python
N = None
pos # [I, J] i j
pos1 = pos[:, :, :, N, N]
pos2 = pos[:, N, N, :, :]
xy_diff = (pos1 - pos2) % image_side  # we make shifts positive by wrapping
attention_bias = gather('i j head , [i, j] i1 j1 i2 j2 -> i1 j1 i2 j2 head', biases, xy_diff)
```
Note that we indeed encounter relative position (shift in x and y), which is not done in most implementations that deal with flat sequence instead.

In a similar way we could produce vector-shift attention (another typical version of relpos):
```python
vector_shift = gather('i j head c, [i, j] i1 j1 i2 j2 -> i1 j1 i2 j2 head c', biases, xy_diff)
```
</details>



## Implementation

Repo provides two implementation:

- array api standard. This implementation is based on a [standard](https://data-apis.org/array-api/latest/) that multiple frameworks pre-agreed to follow.
  Implementation uses only API from standard, so all available operations support all frameworks that follow the standard.

  At some point this should become the one and the only implementation.

  Here is the catch: current support of array api standard is poor, that's why the second implementation exists


- numpy implementation
  
  This independent implementation works right now.

  Numpy implementation is great to test things out, and is handy for a number of non-DL applications as well.


## Development Status

API looks solid, but breaking changes are still possible, so lock the version in your projects (e.g. `eindex==0.1.0`)


## Related projects

Other projects you likely want to look at:

- [tullio](https://github.com/mcabbott/Tullio.jl) by Michael Abbott (@mcabbott) provides Julia macros with a high level of flexibility. 
  Resulting operations are then compiled.
- [torchdim](https://github.com/facebookresearch/torchdim) by Zachary DeVito (@zdevito) introduces "dimension objects", which in particular allow convenient multi-dim indexing
- [einindex](https://github.com/malmaud/einindex) is an einops-inspired prototype by Jonathan Malmaud (@malmaud) to develop multi-dim indexing notation.
  (Also, that's why this package isn't called `einindex`)


## Contributing

We welcome the following contributions:

- next time you deal with multidimensional indexing, do this with eindex <br />
  Worked? &rarr; great - [let us know](https://github.com/arogozhnikov/eindex/discussions/new?category=show-and-tell); didn't work or unclear how to implement &rarr; post in [discussions](https://github.com/arogozhnikov/eindex/discussions)
- if you feel you're already fluent in eindex, help others
- guides/tutorials/video-guides are very welcome, and will be linked
- If you want to translate tutorial to other language and post it somewhere - welcome 



## Discussions

Use github discussions for this project https://github.com/arogozhnikov/eindex/discussions