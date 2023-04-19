---
name: 'Extensions: Introducing new operation / improve notion'
about: 'You want to suggest a new operation or extend an existing one? '
title: "[Feature suggestion]"
labels: feature suggestion
assignees: ''

---

Eindex sparks a lot of interest to introduce new operations that either meet specific requirements or look like a plausible extension of existing ones.

Previous experience with einops shows that most suggestions do not meet, so 

1. Collect **use-cases**. Real ones. Ideas without use-cases are already dead.
2. **Implementation**. (optional) Implementing a sketch of your proposal (e.g. for numpy) allows detecting possible conflicts and realize possible caveats.
3. **Implementation details.** Can you implement this in python array api? If not, what is missing? Is computational graph static (can operation be traced)? 
4. **Integrity** - does it interplay well with existing operations and notation in eindex/einops?
5. **Readability**. This is harder to check, but give it a try. A simple but usable test is to write an exercise sheet with several examples of your extension explained, for others meaning of operation should be guessed. Send this test to a couple of your friends to collect the feedback and see if they can guess encoded meaning. This should also help to improve your ideas to make them more digestible.