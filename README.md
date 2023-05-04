<div align="center">
  <img src="https://arogozhnikov.github.io/images/eindex/logo-animated.gif" alt="eindex animated logo" />
</div>

# eindex

Multidimensional indexing for tensors


## Example of K-means clustering


<table>
<tr>
<td markdown=1>
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
        clstr_indices = np.tile(clusters[:, None], reps=(1, n_dim))
        dim___indices = np.tile(np.arange(n_dim)[None, :], 
                                reps=(n_onservations, 1))
        np.add.at(new_centers_sum, (clstr_indices, dim___indices), X)
        cluster_counts = np.bincount(clusters, minlength=n_clusters)
        centers = new_centers_sum / cluster_counts[:, None]
    return centers
```
</td>
<td markdown=1>
With eindex

```python
def kmeans_eindex(init_centers, X, n_iterations: int):
    centers = init_centers
    for _ in range(n_iterations):
        d = cdist(centers, X)
        clusters = EX.argmin(d, 'cluster i -> [cluster] i')
        centers = EX.scatter('cluster c <- i c, [cluster] i ', X, clusters, 
                             agg='mean', cluster=len(centers))
    return centers








```
</td>
</tr>
</table>

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


## Implementation

Repo provides two implementation:

- array api standard. This implementation is based on a standard that multiple frameworks pre-agreed to follow.
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
- [einindex](https://github.com/malmaud/einindex) is an einops-inspired effort by Jonathan Malmaud (@malmaud) to develop multi-dim indexing notation.
  (Also, that's why this package isn't called `einindex`)


## Contributing

We welcome following contributions:

- next time you deal with multidimensional indexing, do this with eindex <br />
  Worked? &rarr; great - [let us know](https://github.com/arogozhnikov/eindex/discussions/new?category=show-and-tell); didn't work or unclear how to implement &rarr; post in [discussions](https://github.com/arogozhnikov/eindex/discussions)
- if you feel you're already fluent in eindex, help others
- alternative guides/tutorials/video-guides are very welcome
- If you want to translate tutorial to other language and post it somewhere - welcome 



