# Keeping matrices orthogonal

The point here is to make sure, for a given layer, that two inputs aren't being used for the same purpose. If you have two input features being mapped in exactly the same way, that means you're not getting any additional information from one of the features, and they might as well have been combined in a lower level. Therefore, making sure that each input maps in a way that doesn't completely overlap with with another input ensures you transmit maximal information. (NOTE: this is for non-square matrices. For squares, you just initialize orthogonally.)

Orthogonal initialization goes part of the way, making sure that two outputs aren't exactly the same, but it does not ensure that two inputs are treated sufficiently differently. Plus, in a 100x50 matrix, it's clear there are a whole lot of bad orthogonal initializations.

The goal is to initialize your matrices as "incoherent," which means that the rows all have dot products that are bounded by some value. Or, as a looser definition, we want to take all row-dot-products, sum up their absolute values, and minimize this amount. The matrix should be orthogonal in one direction, and as-orthogonal-as-possible in the other.


Hopefully, this provides some inspriration for initializing non-square matrices, which is more of a concern for shallow networks, but doesn't get so much attention.
