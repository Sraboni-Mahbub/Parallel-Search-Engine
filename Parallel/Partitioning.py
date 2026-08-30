# ============================================================
# Partitioning.py
# Smart Partitioning for Heterogeneous Workload
# ============================================================


def estimate_workload(document, number_of_query_words):
    """
    Estimate the computational workload of one document.

    Longer documents require more processing.
    More query words require more searching.

    Workload:
        document length × number of query words
    """

    document_length = len(document)

    workload = (
        document_length *
        max(1, number_of_query_words)
    )

    return workload


# ============================================================
# SMART PARTITIONING
# ============================================================

def smart_partition(
    documents,
    query_words,
    number_of_workers
):
    """
    Divide documents among workers according to
    estimated computational workload.

    This is a greedy workload-balancing algorithm.

    The largest workloads are assigned first to
    the worker with the smallest current workload.
    """

    # --------------------------------------------------------
    # Make sure worker count is valid
    # --------------------------------------------------------

    number_of_workers = max(
        1,
        min(number_of_workers, len(documents))
    )


    # --------------------------------------------------------
    # Calculate workload of every document
    # --------------------------------------------------------

    document_information = []

    for index, document in enumerate(documents):

        workload = estimate_workload(
            document,
            len(query_words)
        )

        document_information.append(
            (
                index,
                document,
                workload
            )
        )


    # --------------------------------------------------------
    # Sort documents from highest workload to lowest
    # --------------------------------------------------------

    document_information.sort(
        key=lambda item: item[2],
        reverse=True
    )


    # --------------------------------------------------------
    # Create partitions
    # --------------------------------------------------------

    partitions = [
        []
        for _ in range(number_of_workers)
    ]


    # Current workload of every worker

    workloads = [
        0
        for _ in range(number_of_workers)
    ]


    # --------------------------------------------------------
    # Assign documents
    # --------------------------------------------------------

    for (
        index,
        document,
        workload
    ) in document_information:

        # Find worker with the smallest workload
        worker_index = workloads.index(
            min(workloads)
        )


        # Assign document
        partitions[worker_index].append(
            (
                index,
                document
            )
        )


        # Update workload
        workloads[worker_index] += workload


    return partitions, workloads