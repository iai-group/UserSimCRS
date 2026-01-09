Dataset
=======

UserSimCRS is shipped with a sample of MovieLens 25M dataset. We specifically chose this dataset as it contains *tags*, which are user-generated metadata about *movies*.
We merge *tags* and *movies* to one file, where each movie in the dataset contains up to 5 most relevant tags. Furthermore, the sampled *ratings* data contains user ratings for 20 users.

The file containing *movies* consists of four headers, separated by ",":

#. movieId
#. title
#. genres (separated by the pipe character)
#. keywords (separated by the pipe character)

Similarly, the *ratings* file consists of four headers, separated by ",":

#. userId
#. movieId
#. rating
#. timestamp

If other datasets are used, we expect them to be in the same format as the *movies* and *ratings* files above.
Note that the *timestamp* field is not in use as of yet. (Can be omitted from the file)

Examples
--------

The following is an entry from the *movies* file:

* 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy,animation|kids and family|pixar animation|computer animation|toys

And an entry from *ratings* file:

* 1,296,5.0,1147880044