- [ ] Composition of activation functions. Can we generate any mathematical function?
- [ ] Do we have enough randomness?
- [ ] Choose a higher mutation rate for individuals which have a worse fitness than
      the avergage.
- [ ] Adapt the mutation rate according to the average behavioral distance
- [ ] Weight mutation: Change until the behavioral distance to the original individual changes by 
      some percentage.  
- [ ] Make probability of structural mutation dependent on the complexity
      (number of nodes, number of links) of the genome.
- [ ] Substrate: Different placement
- [ ] Make weight mutation probability dependent on the current generation
- [ ] Make structural mutation dependent on the average node degree.
      For example, if there is a low connectivity of nodes, adding a new node is
      not a good thing.
- [ ] Add symmetric links, which, when updated, also update their counterpart.
- [ ] Add a fourth objective: Mutation work, which describes how much mutation has happened
      since the beginning for that individual.
- [ ] When adding a link, use a fixed weight for the second link
- [ ] The CPPNs we use, sum all inputs. This way, we cannot
      represent e.g. ```sin(x) * sin(y)```. Add aggregation functions/nodes,
      which can specify arbitrary functions on the inputs.
- [ ] Think about the evolutionary algorithm. Is the current one good enough? Is it correct?
      Do we really want to optimize the diversity?
