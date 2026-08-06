import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Let OpenMP-aware libraries (py4dgeo M3C2, scipy BLAS, numpy BLAS, …) use
# all available cores by default. Was previously pinned to "1" defensively
# alongside KMP_DUPLICATE_LIB_OK; the duplicate-lib flag alone is enough on
# modern stacks, and the OMP=1 pin killed py4dgeo M3C2 parallelism. Override
# downstream by setting OMP_NUM_THREADS in your shell or before `import cntp`
# if you need to cap thread count.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import cntp.coreg
import cntp.io
import cntp.metashape
import cntp.plot
import cntp.preprocess
