# The HDF5 schema tool links conduit::conduit unconditionally, which fails
# the configure when LBANN is built without Conduit. It is a data-preparation
# utility and nothing in the library depends on it.
if (LBANN_HAS_CONDUIT)
  add_subdirectory(hdf5)
endif ()
