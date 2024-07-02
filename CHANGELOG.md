# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Allow no checkboard parameter to be passed in when calling the `DynamicComputeLayer` method directly

## v1.1.6 - 06/06/2024

### Added

- Added reverse colormaps for all available color maps

## v1.1.5 - 05/16/2024

### Changed

- Ensure data is float32 throughout the processing pipeline
- Use private VPCs

## v1.1.4 - 04/25/2024

### Changed

- Update `cloudpickle` for all but python 3.8

### Fixed

- Fixed a bug where reductions over the `bands` axis in ImageStacks could not be visualized

## v1.1.3 - 04/15/2024

### Changed

- Backend changes to support performance during non-peak hours.

## v1.1.1 - 04/12/2024

### Changed

- Bumped cloudpickle from 1.6.0 to 3.0.9 for python 3.11 and later

### Fixed

- Fixed a bug in graft splicing logic
- Fixed a bug when inferring the shape of an empty ImageStack

## v1.1.0 - 03/18/2024

### Added

- Added a resolution operator
- Added an `argmin` reduction

### Changed

- base64 encode numpy arrays using the numpy save functionality instead of pickling them
- Masks are now keyword ops
- Band operations are now keyword ops
- Re-implemented the `dot` operator as a keyword operation.

## v1.0.0 - 03/12/2024

### Changed

- Decreased the use of pickled code and instead utilizing keyword args for most math operations

## v0.9.11 - 02/05/2024

### Changed

- Relaxed the requirement for `descarteslabs` in the client to <3.1.0
- Bumped the requirement for `shapely` in the client to >2
- Return geometries in `ImageStack` as a WKT string instead of a `Polygon` to be compatible with various versions of `shapely`

## v0.9.10 - 01/16/2024

- Backend changes to increase logging and performance

## v0.9.9 - 12/07/2023

### Added

- Expand the client to allow python versions >=3.8, <3.12
- Added the ~ (inversion) operator for Mosaics and ImageStacks

## v0.9.8 - 11/20/2023

### Fixed

- Masking logic ignored prior mask on input.

## v0.9.7 - 11/15/2023

### Changed

- The PixelInspector now shows the lat/lon of the clicked location.
- Added memory optimizations to the backend

## v0.9.6 - 11/08/2023

### Changed

- Backend optimizations to help scalability

## v0.9.5 - 10/31/2023

### Fixed

- Fixed a bug where `dc.dot` did not respect masking.

## v0.9.4 - 10/10/2023

### Fixed

- Fixed a bug that didn't propagate tile padding through graft evaluation correctly.

## v0.9.2 - 09/29/2023

### Changed

- Normalized cache id generation to reduce identical data within a cache

### Fixed

- Addressed a bug where masks are evaluated for 1 by 1 spatial extents.

## v0.9.1 - 09/25/2023

### Fixed

- Retry and then hide "400 Client Error" errors
- Fixed a bug that gave an error if `bands` were passed in as a list of strings to `ImageStack.from_product_bands()`
- Retry and then hide "504 Server Error" errors

### Changed

- Back out some timing measurements that may be causing slowdowns when erroring

## v0.9.0 - 09/20/2023

### Fixed

- Addressed viewing padded mosaics across the anti-meridian.
- Fixed a bug where we expected an error to be a json and it was actually a bytes string
- Addressed a rare chunked-mosaic ordering corner-case.
- Addressed a bug in `pick_bands`
- Removed the ApiCacheError, since not all errors returned here are cache errors, and use `.raise_for_status()` instead of trying to `json.loads()` the error since this was causing issues

### Added

- Nicer error messages if you don't have dynamic-compute-user as one of your groups

### Changed

- Allowing relaxed scale specifications for `Mosaic.visualize`

## v0.8.0 - 08/28/2023

### Added

- Support to promote lists to arrays as graft elements.

### Fixed

- Filtering discarded compute operations, and this has been addressed

## v0.7.0 - 08/09/2023

### Fixed

- Fixed a bug in masking where an ImageStack with one band was not able to be used as a mask

### Changed

- Provide graft optimizations for simple but common patterns that can reduce performance.
- Increase cache timeout
- Forced certifi in the client to be at least the version that addresses a FOSSA vulnerability

## v0.6.4 - 08/01/2023

### Fixed

- Return a user-readable error if the method called on a Mosaic or ImageStack is not supported by `dynamic-compute`

## v0.6.3 - 07/27/2023

### Fixed

- Removed transient errors displayed as tiles

## v0.6.2 - 07/20/2023

### Fixed

- Fixed pixel inspector bug introduced in 0.6.1 that was causing single value reports for multiple band products

## v0.6.1 - 07/19/2023

### Fixed

- Added pixel inspector support for boolean arrays - previously this showed as an explosion image, meaning it failed.

## v0.6.0 - 07/17/2023

### Added

- Added a `compute_all` method to ImageStackGroupBy
- Added a `one` method to ImageStackGroupBy

### Fixed

### Changed

- Upgrade `descarteslabs` dependency to latest version 2.0.3.

## v0.5.0 - 6/27/2023

### Fixed

- Ensured that operations that return ImageStacks that should be filterable, e.g. `pick_bands`, are filterable.
- Fixed a bug that made it so you couldn't visualize Boolean layers

### Changed

- Made frontend and backend changes to record and pass along the python version which then gets used to invoke the correct lambda function.

## v0.4.2 - 6/14/2023

### Changed

- Upgrade `descarteslabs` dependency to latest version 2.0.1.

### Fixed

- Applying a mapped function (such as a reducer) to ImageStackGroupBy now returns an ImageStackGroupBy (which supports saving and loading)
  rather than an ImageStackGroups object (which did not support saving and loading).

## v0.4.1 - 6/13/2023

### Added

- Added `__version__` property to dynamic compute.

### Changed

- Point to PyPI release of `descarteslabs`

## v0.4.0 - 6/8/2023

### Added

- Added support for datetime.datetimes and datetime.date objects when constructing an `ImageStack` or `Mosaic`
- Added support for `.groupby` on `ImageStack`
- Added support for custom reductions (`.reduce()`) on `ImageStack` and `Mosaic`

### Changed

- Consolidated function encoding into a new method
- `ImageStack.visualize()` now throws a descriptive error
- Added tile alerts for when the token has expired
- Moved the colormaps definitions out of the `Mosaic` class and into a file co-located where they are being used
- Refactored the `Mosaic` class to look and operate more like `ImageStack`
- Moved saving and loading of grafts out of ComputeMap and into a catalog module
- Refactored the blob saving/loading functionality to use the new DYNCOMP blob type
- `ImageStack` and `Mosaic` numpy functions such as min, max, median now call the masked version of each function if
  the ImageStack evaluates to a masked array. Also refactored these to be first class functions with docstrings and type hints.

## v0.3.0 - 5/25/2023

### Added

- Added a new dl_utils module in both client and api that adds a get_product_or_fail function
- get_product_or_fail used to replace several instances of the same behavior in api
- get_product_or_fail used to cause earlier indication of bad product access at Mosaic and ImageStack construction time
- New blob module in client.descarteslabs.dynamic_compute library
- New save_to_catalog and load_from_catalog methods on ComputeMap
- Error messages from .visualize now are displayed as tiles

## v0.2.6 - 05/16/2023

### Fixed

- Pixel-wise reductions now pass through properties from the underlying image stack

### Changed

- Relaxed python version requirement for `dynamic-compute`

## v0.2.5 - 05/10/2023

### Changed

- `dynamic-compute` now points to the AWS Catalog

### Fixed

- Fixed some out of date and wrong documentation
- Pixel inspector now works properly with latest dl client

## v0.2.4 - 05/09/2023

### Added

- Added `.mask` functionality to `ImageStack` and added functionality for `Mosaic`s and `ImageStack`s to mask each other.
- Added `pixel` reduction functionality to `ImageStack`

### Fixed

- Fixed zero size array bug that was causing an ugly error on reductions.

## v0.2.3 - 05/08/2023

### Added

- Added Pixel Inspector to DC's Map
- Added Colormap support for single band images on DC's Map
- Added Autoscale capability

### Changed

- Center control now only shows in-bounds coordinates.

### Fixed

- Checkerboard button now works and actually produces checkerboards when appropriate
- Layers can be visualized without scales.

## v0.2.0 - 04/26/2023

### Added

- `zinnia` now handles mosaic requests for empty image collections and returns masked arrays of the appropriate size.
- Added support for grafts that contain arrays, and which evaluate to arrays that are not rasters.
- Expanded the OO API to support masking, clipping.
- Adding support for simple linear algebra.
- Added support for trailing operations, including .pick_bands, .unpack_bands, .rename_bands, and .concat_bands
- Added support for ImageStacks
- Added rudimentary dynamic_compute.map functionality
- Added support for dimensionality reduction over bands in Mosaics, including sum, min, max, median, mean, and argmax
- Added support for viewing reductions over ImageStacks
- Added support for dot operations applied along the "images" axis for ImageStacks.
- Added .pick_bands, .rename_bands, .unpack_bands, and .concat_bands to ImageStacks
- Added functionality for scaling interactive map layers
- Added support for binary operations where one operand is an ImageStack and the other is a Mosaic
- .compute now support DLTIles and XYZTiles as well as AOIs

### Changed

- Bumped the DL client version from 1.11 to 1.12.1 and moved from `dl.scenes` to `dl.catalog` and `dl.geo`.
- Compute tile operations now return a detailed error message on failure
- Calls to Mosaic now return arrays in physical, rather than data, range.
- Changed default `zinnia` deployment to be in `appsci-production`
- Changed to name from `zinnia` to `dynamic-compute`
- Changed how we handle caching of proxy objects -- all grafts for Mosaics or ImageStacks are set as cacheable.
- `dynamic_compute` is now imported as `descarteslabs.dynamic_compute`

### Fixed

- Addressed an error that caused map layers to be re-added instead of replaced when appropriate.
- Addressed how we handled image stacks of length zero.
- Addressed an error that caused out of bounds bboxes to be passed to an AOI object in Map().geocontext()
