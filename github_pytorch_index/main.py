#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if __name__ == "__main__":
    import sys
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)

    from paritybench.main import main
    main()
