
from typing import List, Tuple
import torch

import functools


def has_cuda_cudart() -> bool:
    """
    Check if cuda.cudart module is available (cuda-python <= 12.9).

    Returns:
        True if cuda.cudart exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("cuda.cudart") is not None


try:
    import cuda.bindings.driver as driver
    import cuda.bindings.runtime as runtime
    from cuda.bindings.driver import CUdevice, CUdevResource
except ImportError as e:
    raise ImportError(
        "Could not import the 'cuda' module. "
        "Please install cuda-python that matches your CUDA version."
    ) from e

try:
    # Check if cuda.cudart module is available and import accordingly
    if has_cuda_cudart():
        # cuda-python < 12.9 (has cuda.cudart)
        import cuda.bindings.driver as driver
        import cuda.bindings.runtime as runtime
        import cuda.cudart as cudart
        import cuda.nvrtc as nvrtc
    else:
        # cuda-python >= 13.0 (no cuda.cudart, use runtime as cudart)
        from cuda.bindings import driver, nvrtc, runtime

        cudart = runtime  # Alias runtime as cudart for compatibility
except ImportError as e:
    raise ImportError(
        "Could not import the 'cuda' module. "
        "Please install cuda-python that matches your CUDA version."
    ) from e

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, runtime.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]



def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y"""
    return ceil_div(x, y) * y



@functools.cache
def get_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        raise ValueError("device must be a cuda device")
    return torch.cuda.get_device_capability(device.index)


def get_sm_count_constraint(major: int, minor: int) -> Tuple[int, int]:
    if major == 6:
        return (1, 1)
    elif major == 7:
        return (2, 2)
    elif major == 8:
        return (4, 2)
    elif major >= 9:
        return (8, 8)
    else:
        raise ValueError(f"Unsupported CUDA capability: {major}.{minor}")


def get_cudevice(dev: torch.device) -> CUdevice:
    try:
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    except RuntimeError:
        runtime.cudaInitDevice(dev.index, 0, 0)
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    return cu_dev


def get_device_resource(cu_dev: CUdevice) -> CUdevResource:
    return checkCudaErrors(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )


def split_resource(
    resource: CUdevResource,
    num_groups: int,
    min_count: int,
) -> Tuple[CUdevResource, CUdevResource]:
    results, _, remaining = checkCudaErrors(
        driver.cuDevSmResourceSplitByCount(
            num_groups,
            resource,
            0,  # useFlags
            min_count,
        )
    )
    return results, remaining


def split_resource_by_sm_count(
    cu_dev: CUdevice, resource: CUdevResource, sm_counts: List[int]
) -> Tuple[List[CUdevResource], CUdevResource]:
    results = []
    for sm_count in sm_counts:
        result, remaining = split_resource(resource, 1, sm_count)
        results.extend(result)
        # Refresh the remaining resource for the next iteration
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([remaining], 1))
        green_ctx = checkCudaErrors(
            driver.cuGreenCtxCreate(
                desc, cu_dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        resource = checkCudaErrors(
            driver.cuGreenCtxGetDevResource(
                green_ctx, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
        )

    return results, resource


def create_green_ctx_streams(
    cu_dev: CUdevResource, resources: List[CUdevResource]
) -> List[torch.Stream]:
    streams = []
    for split in resources:
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([split], 1))
        green_ctx = checkCudaErrors(
            driver.cuGreenCtxCreate(
                desc, cu_dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        stream = checkCudaErrors(
            driver.cuGreenCtxStreamCreate(
                green_ctx,
                driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
                0,  # priority
            )
        )
        streams.append(torch.cuda.get_stream_from_external(stream))

    return streams


def split_device_green_ctx(
    dev: torch.device, num_groups: int, min_count: int
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Split the device into multiple `green contexts <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>`_,
    return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
    Green contexts allow concurrent execution of multiple kernels on different SM partitions.

    Args:
        dev: The device to split.
        num_groups: The number of groups to split the device into.
        min_count: Minimum number of SMs required for each group, it will be adjusted to meet the
            alignment and granularity requirements.

    Returns:
        streams: The list of torch.Streams objects corresponding to the green contexts.
        resources: The list of CUdevResource objects corresponding to the green contexts.

    Example:
        >>> from flashinfer.green_ctx import split_device_green_ctx
        >>> import torch
        >>> dev = torch.device("cuda:0")
        >>> streams, resources = split_device_green_ctx(dev, 2, 16)
        >>> print([r.sm.smCount for r in resources])
        [16, 16, 100]
        >>> with torch.cuda.stream(streams[0]):
        ...     x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     z = x @ y
        ...     print(z.shape)
        ...
        torch.Size([8192, 8192])

    Note:
        The length of the returned streams and resources is ``num_groups + 1``,
        where the last one is the remaining SMs.

        The following examples show how the SM count is rounded up to meet the alignment and granularity requirements:
        - Requested 7 SMs → Allocated 8 SMs (rounded up to minimum)
        - Requested 10 SMs → Allocated 16 SMs (rounded up to multiple of 8)
        - Requested 16 SMs → Allocated 16 SMs (no rounding needed)
        - Requested 17 SMs → Allocated 24 SMs (rounded up to multiple of 8)

    Raises:
        RuntimeError: when requested SM allocation exceeds device capacity:
        ``num_groups * rounded_min_count > total_device_sms``
    """
    try:
        cu_dev = get_cudevice(dev)
        resource = get_device_resource(cu_dev)
        results, remaining = split_resource(resource, num_groups, min_count)
        resources = results + [remaining]
        streams = create_green_ctx_streams(cu_dev, resources)
        return streams, resources
    except RuntimeError as e:
        if (
            "CUDA error code=914" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_TYPE" in str(e)
            or "CUDA error code=915" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION" in str(e)
        ):
            raise RuntimeError(
                f"{e}\n"
                f"Failed to split device into {num_groups} groups with min_count={min_count}. "
                f"This is likely due to insufficient number of SMs available on the device. "
                f"Please reduce the number of groups or the minimum SM count per group."
            ) from e
        raise


def split_device_green_ctx_by_sm_count(
    dev: torch.device, sm_counts: List[int]
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Split the device into multiple green contexts, each with a fixed number of SMs,
    return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
    Green contexts allow concurrent execution of multiple kernels on different SM partitions.

    Args:
        dev: The device to split.
        sm_counts: List of SM counts for each partition. Each count will be rounded up
                   to meet the minimum and alignment requirements.

    Returns:
        streams: The list of torch.Streams objects corresponding to the green contexts.
        resources: The list of CUdevResource objects corresponding to the green contexts.

    Raises:
        RuntimeError: If the requested SM allocation exceeds device capacity:
            - When sum(rounded_sm_counts) > total_device_sms
            - When CUDA operations fail due to invalid resource types
            - When the device is not properly initialized
        ValueError: If sm_counts is empty or contains invalid values (e.g., negative values).

    Example:
        >>> from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
        >>> import torch
        >>> dev = torch.device("cuda:0")
        >>>
        >>> # Create three partitions with specific SM counts
        >>> streams, resources = split_device_green_ctx_by_sm_count(dev, [8, 16, 24])
        >>> print([r.sm.smCount for r in resources])
        [8, 16, 24, 84]  # Last value is remaining SMs
        >>>
        >>> # Execute kernels on different partitions
        >>> with torch.cuda.stream(streams[0]):
        ...     x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        ...     y = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        ...     z = x @ y
        ...     print(f"Partition 0 result: {z.shape}")
        ...
        >>> with torch.cuda.stream(streams[1]):
        ...     # Different computation on partition 1
        ...     a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        ...     b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        ...     c = a @ b
        ...     print(f"Partition 1 result: {c.shape}")

    Note:
        The length of the returned streams and resources is ``len(sm_counts) + 1``,
        where the last one contains the remaining SMs that were not allocated.

        SM count alignment examples for Compute Capability 9.0+:
        - Requested 7 SMs → Allocated 8 SMs (rounded up to minimum)
        - Requested 10 SMs → Allocated 16 SMs (rounded up to multiple of 8)
        - Requested 16 SMs → Allocated 16 SMs (no rounding needed)
        - Requested 17 SMs → Allocated 24 SMs (rounded up to multiple of 8)

        The actual SM count can be obtained from the ``.sm.smCount`` field of the returned resources.

        See `CUDA Green Contexts <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>`_
        for more details.
    """
    try:
        cu_dev = get_cudevice(dev)
        resource = get_device_resource(cu_dev)

        # Round sm counts to meet the alignment and granularity requirements
        rounded_sm_counts = []
        for sm_count in sm_counts:
            min_sm_count, sm_alignment = get_sm_count_constraint(
                *get_compute_capability(dev)
            )
            if sm_count <= 0:
                raise ValueError(f"SM count must be positive, got {sm_count}")
            rounded_sm_counts.append(
                round_up(max(sm_count, min_sm_count), sm_alignment)
            )

        # Split the device into multiple green contexts
        results, remaining = split_resource_by_sm_count(
            cu_dev, resource, rounded_sm_counts
        )
        resources = results + [remaining]
        streams = create_green_ctx_streams(cu_dev, resources)
        return streams, resources
    except RuntimeError as e:
        if (
            "CUDA error code=914" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_TYPE" in str(e)
            or "CUDA error code=915" in str(e)
            or "CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION" in str(e)
        ):
            raise RuntimeError(
                f"{e}\n"
                f"Failed to split device with SM counts {sm_counts} (rounded to {rounded_sm_counts}). "
                f"This is likely due to insufficient number of SMs available on the device. "
                f"Please reduce the requested SM counts or use fewer partitions."
            ) from e
        raise