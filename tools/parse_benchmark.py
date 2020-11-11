#!/usr/bin/python3

import csv
import statistics
import sys

benchmark_data = {}
normalizers = {
    'second': 1.0,
    's': 1.0,
    'millisecond': 1 / 1000,
    'ms': 1 / 1000,
    'microsecond': 1 / 1000 / 1000,
    'us': 1 / 1000 / 1000,
    'nanosecond': 1 / 1000 / 1000 / 1000,
    'ns': 1 / 1000 / 1000 / 1000,
}

def parse_raw_benchmark_line(line):
    info = line.rstrip().split("sb: benchmark: ")[1]
    # pray that there aren't spaces in the file path
    kvs = info.split()

    kvs = [kv.partition('=')[::2] for kv in kvs]
    kvs = [(kv[0], kv[1].strip('\'')) for kv in kvs]
    d = dict(kvs)

    try:
        return d["file"], d["func"], float(d["time"]), d["units"]
    except:
        return None, None, None, None

def add_datapoint(srcfile, function, time, units):
    global benchmark_data
    if not srcfile in benchmark_data:
        benchmark_data[srcfile] = {}
    if not function in benchmark_data[srcfile]:
        benchmark_data[srcfile][function] = []
    benchmark_data[srcfile][function].append((time, units))

def is_benchmark_line(line):
    return line.startswith("sb: benchmark: ")

try:
    for line in sys.stdin:
        if is_benchmark_line(line):
            srcfile, function, time, units = parse_raw_benchmark_line(line)
            if not srcfile is None:
                add_datapoint(srcfile, function, time, units)
except KeyboardInterrupt:
    pass

results = []

for ks in benchmark_data:
    srcfile = benchmark_data[ks]
    for kf in srcfile:
        try:
            func = srcfile[kf]
            first_unit = func[0][1]
            denormalizer = 1 / normalizers[first_unit]
            values_only = [normalizers[d[1]] * d[0] for d in func]
            m = statistics.mean(values_only) * denormalizer
            s = statistics.stdev(values_only) * denormalizer
            dmin = min(values_only) * denormalizer
            dmax = max(values_only) * denormalizer
            total = sum(values_only) * denormalizer
            results.append([ks, kf, m, s, dmin, dmax, total, first_unit])
        except statistics.StatisticsError as ex:
            pass

w = csv.writer(sys.stdout)
print("file,func,mean,stdev,min,max,total,unit")
w.writerows(results)
