#!/usr/bin/python3

import sys
import os.path
import csv

if len(sys.argv) < 2:
    print("Usage: {} path_to_data_dir".format(sys.argv[0]))
    sys.exit(1)

data_dir = sys.argv[1]

for backend in range(1, 4):
    outputs = [
        ["step{}.csv", "step", None, None],
        ["shapematch{}.csv", "do_one_iteration_of_shape_matching_constraint_resolution", None, None],
        ["predict{}.csv", "predict", None, None],
    ]

    for output in outputs:
        f = open(os.path.join(data_dir, output[0].format(backend)), "w")
        w = csv.writer(f, delimiter=',')
        output[2] = f
        output[3] = w
        w.writerow(['size', 'avg', 'stdev', 'unit'])

    for size in range(4, 115):
        srcpath = "stats.{}.{}.csv".format(backend, size)
        with open(os.path.join(data_dir, srcpath)) as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_func = header.index("func")
            idx_mean = header.index("mean")
            idx_stdev = header.index("stdev")
            idx_unit = header.index("unit")
            for row in reader:
                mean = row[idx_mean]
                stdev = row[idx_stdev]
                unit = row[idx_unit]
                for output in outputs:
                    if row[idx_func] == output[1]:
                        output[3].writerow([size, mean, stdev, unit])
                        break

    for output in outputs:
        output[2].close()
