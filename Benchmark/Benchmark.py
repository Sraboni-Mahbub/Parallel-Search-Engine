from pathlib import Path


# ============================================================
# PROJECT ROOT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# FILE PATHS
# ============================================================

SEQUENTIAL_FILE = (
    BASE_DIR
    / "Dataset2"
    / "sequential_time.txt"
)


PARALLEL_FILE = (
    BASE_DIR
    / "Dataset2"
    / "parallel_time.txt"
)


# ============================================================
# CHECK SEQUENTIAL FILE
# ============================================================

if not SEQUENTIAL_FILE.exists():

    print(
        "ERROR: sequential_time.txt not found."
    )

    print(
        "Expected location:"
    )

    print(
        SEQUENTIAL_FILE
    )

    exit()


# ============================================================
# CHECK PARALLEL FILE
# ============================================================

if not PARALLEL_FILE.exists():

    print(
        "ERROR: parallel_time.txt not found."
    )

    print(
        "Expected location:"
    )

    print(
        PARALLEL_FILE
    )

    exit()


# ============================================================
# READ SEQUENTIAL DATA
# ============================================================

with open(
    SEQUENTIAL_FILE,
    "r"
) as file:

    sequential_content = (
        file.read().strip()
    )


if not sequential_content:

    print(
        "ERROR: sequential_time.txt is empty."
    )

    print(
        "Please run Sequential.py again."
    )

    exit()


try:

    sequential_time = float(
        sequential_content
    )

except ValueError:

    print(
        "ERROR: Invalid sequential time."
    )

    print(
        "File contains:"
    )

    print(
        repr(sequential_content)
    )

    exit()


# ============================================================
# READ PARALLEL DATA
# ============================================================

with open(
    PARALLEL_FILE,
    "r"
) as file:

    lines = [

        line.strip()

        for line in file

        if line.strip()

    ]


# ============================================================
# CHECK PARALLEL DATA
# ============================================================

if len(lines) < 2:

    print(
        "ERROR: parallel_time.txt does not contain "
        "both execution time and worker count."
    )

    print(
        "Expected format:"
    )

    print(
        "0.774297"
    )

    print(
        "4"
    )

    print(
        "\nActual file content:"
    )

    print(
        repr(lines)
    )

    exit()


# ============================================================
# CONVERT PARALLEL DATA
# ============================================================

try:

    parallel_time = float(
        lines[0]
    )


    workers = int(
        lines[1]
    )


except ValueError:

    print(
        "ERROR: Invalid data inside parallel_time.txt."
    )

    print(
        "Actual file content:"
    )

    print(
        repr(lines)
    )

    exit()


# ============================================================
# CALCULATE SPEEDUP
#
# Speedup = Sequential Time / Parallel Time
# ============================================================

speedup = (

    sequential_time
    / parallel_time

)


# ============================================================
# CALCULATE EFFICIENCY
#
# Efficiency = Speedup / Number of Workers
# ============================================================

efficiency = (

    speedup
    / workers

)


efficiency_percentage = (

    efficiency
    * 100

)


# ============================================================
# CALCULATE PERFORMANCE IMPROVEMENT
# ============================================================

performance_improvement = (

    (
        sequential_time
        - parallel_time
    )

    / sequential_time

) * 100


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "PARALLEL SEARCH BENCHMARK RESULTS"
)

print(
    "=========================================="
)


print(
    f"\nSequential Time: "
    f"{sequential_time:.6f} seconds"
)


print(
    f"Parallel Time: "
    f"{parallel_time:.6f} seconds"
)


print(
    f"Number of Workers: "
    f"{workers}"
)


print(
    "\n------------------------------------------"
)


print(
    f"Speedup: "
    f"{speedup:.4f}x"
)



print(
    f"Efficiency Percentage: "
    f"{efficiency_percentage:.2f}%"
)



print(
    "\n=========================================="
)