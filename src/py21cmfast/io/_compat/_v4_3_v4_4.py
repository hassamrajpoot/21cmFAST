"""Changes to the output format between v4.3.0 and v4.4.0."""

from .._compat_framework import AddArrayMetadata, FormatRelease

RELEASE = FormatRelease("4.3.0", "4.4.0", [AddArrayMetadata()])
