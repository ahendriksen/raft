# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from pathlib import Path
from collections import Counter

# The ninja log file format is quite simple. For more info, see:
# https://github.com/m-ou-se/ninj/issues/13
def parse_ninja_log(log_path):
    text = Path(log_path).read_text()
    start, end, mtime, path, cmd = list(zip(*[line.split("\t") for line in text.splitlines()[1:]]))
    start = list(map(int, start))
    end = list(map(int, end))
    seconds = [(e - s) / 1000. for e, s in zip(end, start)]
    mtime = list(map(int, mtime))

    return dict(
        start=start,
        end=end,
        seconds=seconds,
        mtime=mtime,
        path=path,
        cmd=cmd
    )

def discard_earlier_builds(d):
    prev_end = 0
    start_index = 0
    # end must be monotonically increasing. If we find and end value that is
    # lower than the end value on the previous row, we know that a new build has
    # started.
    for i, end in enumerate(d['end']):
        if end < prev_end:
            start_index = i
        prev_end = end

    return {k: v[start_index:] for k, v in d.items()}


def main(ninja_log_path):
    log = discard_earlier_builds(parse_ninja_log(ninja_log_path))
    times = sorted(zip(log['path'], log['seconds']), key=lambda x: x[1])

    for p, s in times:
        print(f"{p[-120:]:<120} {s:6.1f} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""analyze-ninja-log.py

Prints build times of each step of the last ninja build. The output is ordered
from shortest to longest build time.

Usage:

    python analyze-ninja-log.py path/to/.ninja_log
        """)
        exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Path {input_path} does not exist.")
    else:
        main(input_path)
