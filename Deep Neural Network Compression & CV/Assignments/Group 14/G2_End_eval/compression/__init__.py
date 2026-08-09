"""
compression/
────────────
Layer wrappers + compression algorithms.

  modified_linear   – drop-in nn.Linear with mask + KMeans support
  modified_conv2d   – drop-in nn.Conv2d with mask + KMeans support
  pruning           – global magnitude pruning
  structured_pruning– filter-level structured pruning (conv layers)
  ptq               – post-training dynamic / static int8 quantisation
  sparse            – CSR sparse encoding for actual disk-size reduction
"""
