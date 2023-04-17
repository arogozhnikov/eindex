# eindex

Multidimensional indexing for tensors


## Goals

- Form helpful 'language' to think about indexing and index-related operations. Tools shape mind 
- Cover most common cases of multidimensional indexing that are hard to implement using the standard API
- Approach should be applicable to most common tensor frameworks, autograd should work out-of-the-box
- Aim for readable and reliable code
- Allow simple adoption in existing codebases
- Implementation should base on fairly common tensor operations. No custom kernels allowed.
- Desired limitation: execution plan for every operation should form a static graph. 
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
  There is no optimization right now.


## Development Status

API looks solid, but breaking changes are still possible, so lock the version in your projects (e.g. `eindex==0.1.0`)


## Related projects

Other projects you likely want to look at:

- [tullio](https://github.com/mcabbott/Tullio.jl) by Michael Abbott (@mcabbott) provides Julia macros with a high level of flexibility. 
  Resulting operations are then compiled.
- [torchdim](https://github.com/facebookresearch/torchdim) by Zachary DeVito (@zdevito) introduces "dimension objects", which in particular allow convenient multi-dim indexing
- [einindex](https://github.com/malmaud/einindex) by 
is an einops-inspired effort by Jonathan Malmaud (@malmaud) to develop multi-dim indexing notation.
  (Also, that's why this package isn't called `einindex`)


## Contributing

Minimization of maintenance costs is critical.

Right ways to contribute:

- next time you deal with multidimensional indexing, do this with eindex <br />
  Worked? -> great - [let us know](https://github.com/arogozhnikov/eindex/discussions/new?category=show-and-tell); didn't work or unclear how to implement -> post in [discussions](https://github.com/arogozhnikov/eindex/discussions)
- if you feel you're already fluent in eindex, help others
- alternative guides/tutorials/video-guides are very welcome
- If you want to translate tutorial to other language and post it somewhere - welcome 


Wrong way to contribute: suggesting more operations/features.

There is an infinite space of operations, and a long list of requirements / desired properties for operations.
Thus in design of operations I prefer detailed boring analysis of usecases to exciting feature-packing. 
