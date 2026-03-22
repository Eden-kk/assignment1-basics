First, pre-tokenization determines the boundary of tokenization, in other word, two bytes that are next to each other but belong to 2 different pre-token cannot be merged during the training process.

The vocabulary we'll get is a mapping froms raw byte sequence to token(index)

Next, the training process use the frequency from pairs of all of the pretokens and pick the next pair with highest frequency to merge, while the encoding process faces with each token and encodes them separately, which could be paralleled.

Besides, during the encoding process, what we repeatedly traverse is the learned merged list rather than current pre-token that is being encoded: everytime after merging a new pair of token to a new token, we update the pre-token's representation with the new token, and re-traverse the learned merged list to find out next potentially possible pair that can be merged.

When using encode_iterable(), be careful about the tail of each chunk as they may be an incomplete token's bytes; while `deferred_suffix` prevents from special token to be truncated, `previous_pretoken` prevents a longer token be truncated to two shorter tokens, which will be concatenated with `deferred_suffix` to form a complete token.

After pretokenization, record the mapping from each unique pair in the pretokens to pair count and pretokens that include them, which can accelerate the merging process:
- before building this mapping, in each merging loop, the total pretokens need to be scanned and record pairs and their frequency
- after building this mapping, maintaining the mappings is enough, the total pretokens no longer need to be considered

Pretokens are important, as tokens are split inside pretokens 